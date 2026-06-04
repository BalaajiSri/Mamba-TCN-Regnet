"""
Financial-grade loss functions for HybridSSM-Trader.

References
----------
GMADL  : Sakowski et al. (2024) arxiv:2412.18405
          "Generalized Mean Absolute Directional Loss as a Solution to
          Overfitting and High Transaction Costs in Machine Learning Models
          Used in High-Frequency Algorithmic Investment Strategies"

MADL   : Michańków et al. (2023) arxiv:2309.10546
          "Mean Absolute Directional Loss as a New Loss Function for ML
          Problems in Algorithmic Investment Strategies"

Focal  : Lin et al. (2017) "Focal Loss for Dense Object Detection"

EMD    : Earth Mover's Distance for ordinal classification
         (Clinically Aware Learning, Sluijterman et al. 2024)

Sharpe : Differentiable Sharpe ratio auxiliary (standard in deep RL trading)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Regression losses ────────────────────────────────────────────────────────


class GMAdaptiveLoss(nn.Module):
    """Generalized Mean Absolute Directional Loss (GMADL).

    GMADL(ŷ, y; α, β, τ) = E[ |ŷ_τ|^α · (1 − σ(β · y · ŷ_τ)) ]

    where  ŷ_τ = sign(ŷ) · max(|ŷ| − τ, 0)  soft-thresholds the signal by the
    transaction cost τ, so predictions smaller than the round-trip cost are
    treated as zero (no signal).

    Parameters
    ----------
    alpha : float
        Magnitude exponent.  α=1 recovers MADL-like behaviour;
        α→0 gives a pure directional loss.
    beta  : float
        Sharpness of the sigmoid direction gate.  Higher β gives a steeper
        transition around the correct-direction boundary.
    tau   : float
        Transaction cost threshold.  Predictions with |ŷ| < τ contribute
        no loss, discouraging low-confidence trades that don't clear costs.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 10.0, tau: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.tau = tau

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y_pred : (B,) or (B, 1)   predicted log-returns
        y_true : (B,) or (B, 1)   actual log-returns
        """
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)

        # Soft-threshold by transaction cost τ
        if self.tau > 0.0:
            y_hat = torch.sign(y_pred) * F.relu(y_pred.abs() - self.tau)
        else:
            y_hat = y_pred

        # |ŷ_τ|^α  — magnitude weight
        magnitude = y_hat.abs().clamp_min(1e-8).pow(self.alpha)

        # 1 − σ(β · y · ŷ_τ)  — direction penalty
        #   → 0  when prediction and truth agree (correct direction)
        #   → 1  when they disagree (wrong direction)
        direction_penalty = 1.0 - torch.sigmoid(self.beta * y_true * y_hat)

        return (magnitude * direction_penalty).mean()


class DifferentiableSharpe(nn.Module):
    """Differentiable negative Sharpe ratio auxiliary loss.

    Uses tanh(ŷ / T) as a smooth proxy for sign(ŷ) to form a differentiable
    position, then measures the mean/std of the implied period returns.

    Minimising this loss directly pushes the model toward strategies with
    higher risk-adjusted returns, complementing the direction-focused GMADL.

    Parameters
    ----------
    temperature : float
        Controls how quickly tanh saturates.  Smaller T → harder sign;
        larger T → softer, more conservative positions.
    eps : float
        Floor on the standard deviation to prevent division by zero.
    """

    def __init__(self, temperature: float = 0.02, eps: float = 1e-6):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y_pred : (B,)  predicted log-returns
        y_true : (B,)  actual log-returns for the same periods
        """
        y_pred = y_pred.reshape(-1)
        y_true = y_true.reshape(-1)

        # Differentiable position in (−1, +1)
        position = torch.tanh(y_pred / self.temperature)

        # Implied realised return for each period
        realized = position * y_true

        mean_r = realized.mean()
        std_r = realized.std(unbiased=False).clamp_min(self.eps)

        # Return *negative* Sharpe so we can minimise
        return -(mean_r / std_r)


# ─── Action / classification losses ──────────────────────────────────────────


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced multi-class classification.

    FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)

    Down-weights well-classified 'hold' examples so the gradient is dominated
    by the harder, rarer buy/sell signals.

    Parameters
    ----------
    gamma      : float  Focusing parameter.  γ=0 → standard CE; γ=2 typical.
    class_weight : (C,) tensor or None
                  Per-class weight α_t.  If None, uniform.
    """

    def __init__(self, gamma: float = 2.0, class_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("class_weight", class_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard cross-entropy (per sample, unreduced)
        ce = F.cross_entropy(logits, targets, weight=self.class_weight, reduction="none")
        # p_t = exp(−CE) is the probability assigned to the true class
        p_t = torch.exp(-ce)
        focal_weight = (1.0 - p_t).pow(self.gamma)
        return (focal_weight * ce).mean()


class OrdinalEMDLoss(nn.Module):
    """Earth Mover's Distance loss for ordinal buy/sell/hold.

    EMD(p, q) = Σ_k |CDF_p(k) − CDF_q(k)|

    Because sell < hold < buy, a buy↔sell mistake crosses two CDF steps and
    incurs twice the penalty of a buy↔hold or hold↔sell mistake.  This is
    strictly better than cross-entropy for an ordinal signal.

    Parameters
    ----------
    num_classes : int  Number of ordinal classes (3 for sell/hold/buy).
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)                                  # (B, C)
        target_probs = F.one_hot(targets, self.num_classes).float()        # (B, C)

        # CDFs (the last element is always 1 for both, so drop it)
        cdf_pred = torch.cumsum(probs, dim=-1)[:, :-1]                     # (B, C−1)
        cdf_true = torch.cumsum(target_probs, dim=-1)[:, :-1]             # (B, C−1)

        emd = (cdf_pred - cdf_true).abs().sum(dim=-1)                     # (B,)
        return emd.mean()


class ReturnWeightedCE(nn.Module):
    """Cross-entropy weighted by the magnitude of each sample's actual return.

    Near-threshold 'hold' labels (tiny returns) barely contribute to the loss;
    large-move days where the model must decide correctly dominate.

    Parameters
    ----------
    eps : float  Small floor so zero-return samples aren't completely ignored.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C)
        targets : (B,)   integer class labels
        returns : (B,)   actual returns for the period (used as weights)
        """
        ce = F.cross_entropy(logits, targets, reduction="none")            # (B,)
        weights = returns.abs().clamp_min(self.eps)
        weights = weights / weights.mean()                                  # normalise
        return (weights * ce).mean()


# ─── Auxiliary regularisation ─────────────────────────────────────────────────


class TurnoverPenalty(nn.Module):
    """Penalises rapid position changes between consecutive samples.

    Computes expected position from soft action probabilities and penalises
    the L1 distance between adjacent expected positions in the batch.

    Assumes the batch is ordered temporally (no shuffling), or that at minimum
    the penalty provides a useful smoothness prior even if ordering is broken.

    action_to_pos : mapping from class indices to positions
        default: sell=−1, hold=0, buy=+1
    """

    def __init__(self, action_to_pos: list[float] | None = None):
        super().__init__()
        self.action_to_pos = action_to_pos or [-1.0, 0.0, 1.0]

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[0] < 2:
            return logits.sum() * 0.0  # differentiable zero

        probs = F.softmax(logits, dim=-1)                                  # (B, C)
        pos_map = torch.tensor(
            self.action_to_pos, dtype=logits.dtype, device=logits.device
        )
        # Expected position for each sample
        expected_pos = (probs * pos_map).sum(dim=-1)                       # (B,)
        changes = (expected_pos[1:] - expected_pos[:-1]).abs()             # (B−1,)
        return changes.mean()
