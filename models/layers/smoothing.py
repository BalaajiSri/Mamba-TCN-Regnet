from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveSignalSmoother(nn.Module):
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
        volatility = self._estimate_volatility(raw_inputs, close_index=close_index)
        alpha = self.volatility_gate(volatility)
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * alpha

        smoothed_steps = [states[:, 0, :]]
        for step in range(1, states.shape[1]):
            prev = smoothed_steps[-1]
            current = states[:, step, :]
            smoothed_steps.append(alpha * current + (1.0 - alpha) * prev)
        return torch.stack(smoothed_steps, dim=1)

