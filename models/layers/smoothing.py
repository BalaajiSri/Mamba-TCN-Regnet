from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveSignalSmoother(nn.Module):
    """Volatility-gated exponential smoothing over the hidden sequence.

    The smoothing factor α is predicted from the current-period volatility of
    the close price.  High volatility → α closer to max_alpha (more weight on
    the current step); low volatility → α closer to min_alpha (more smoothing).

    Vectorised EMA
    ──────────────
    Because α is constant over time for a given batch element (it depends only
    on whole-window volatility, not per-step), the EMA can be computed without
    a Python loop using prefix-weighted sums:

        out[t] = Σ_{k=0}^{t}  α · (1−α)^{t−k} · x[k]  +  (1−α)^{t+1} · x[0]

    Rearranged as a matrix-vector product with a causal decay matrix, then
    computed efficiently via reverse cumsum of log-decay terms.
    """

    def __init__(
        self,
        hidden_dim: int,
        min_alpha: float = 0.15,
        max_alpha: float = 0.85,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.eps = eps
        self.volatility_gate = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Sigmoid(),
        )

    def _estimate_volatility(self, raw_inputs: torch.Tensor, close_index: int) -> torch.Tensor:
        close = raw_inputs[:, close_index, :]
        returns = torch.diff(close, dim=-1) / close[:, :-1].clamp_min(self.eps)
        volatility = returns.std(dim=-1, keepdim=True, unbiased=False)
        return volatility

    def forward(self, states: torch.Tensor, raw_inputs: torch.Tensor, close_index: int = 3) -> torch.Tensor:
        """
        Args:
            states:     (B, T, D)  hidden states to smooth.
            raw_inputs: (B, F, T)  original input features (for volatility est.)
            close_index: column index of the close price in raw_inputs.

        Returns:
            Smoothed states of shape (B, T, D).
        """
        volatility = self._estimate_volatility(raw_inputs, close_index=close_index)
        # alpha: (B, 1, D) — one smoothing factor per batch element & dimension.
        alpha = self.volatility_gate(volatility)                         # (B, D)
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * alpha
        alpha = alpha.unsqueeze(1)                                        # (B, 1, D)

        B, T, D = states.shape
        decay = 1.0 - alpha  # (B, 1, D)

        # Build causal decay exponents: at time-step t, position k contributes
        # α·decay^(t-k).  We compute this via outer subtraction in log-space.
        t_idx = torch.arange(T, device=states.device, dtype=states.dtype)  # (T,)
        # diff[t, k] = t - k  (lower-triangular, negative for k > t)
        diff = t_idx.unsqueeze(0) - t_idx.unsqueeze(1)                    # (T, T)
        # Causal mask: upper triangle gets weight 0.
        causal_mask = diff >= 0                                            # (T, T)

        # log_decay: (B, 1, D) → broadcast to (B, T, T, D) via einsum is heavy.
        # Instead, work per-dimension: decay is shared across T so we can
        # separate the computation.
        #
        # weights[b, t, k, d] = α[b,d] · decay[b,d]^(t-k)   for k ≤ t
        #                      = 0                              for k > t
        #
        # Since decay is the same for all (t, k), we only need decay^diff.
        # We compute this as exp(diff * log(decay)) with a clamp for safety.
        log_decay = torch.log(decay.clamp_min(1e-7))                     # (B, 1, D)
        # diff: (T, T) → (1, T, T, 1); log_decay: (B, 1, 1, D)
        exponents = diff.unsqueeze(0).unsqueeze(-1) * log_decay.unsqueeze(2)  # (B, T, T, D)
        weights = alpha.unsqueeze(2) * torch.exp(exponents)               # (B, T, T, D)
        # Zero out non-causal positions.
        weights = weights * causal_mask.unsqueeze(0).unsqueeze(-1)        # (B, T, T, D)

        # Weighted sum: out[b, t, d] = Σ_k weights[b, t, k, d] · states[b, k, d]
        # states: (B, T, D) → (B, 1, T, D); weights: (B, T, T, D)
        smoothed = (weights * states.unsqueeze(1)).sum(dim=2)             # (B, T, D)
        return smoothed
