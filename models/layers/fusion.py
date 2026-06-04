from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedFeatureFusion(nn.Module):
    """Three-way gated fusion of TCN (local), Mamba (long-range), and attention
    (global) context streams.

    A softmax gate over the concatenated projections produces per-branch mixture
    weights that always sum to 1, so each branch competes fairly for influence
    instead of the old broken formulation where the global branch always
    contributed a fixed 0.5 offset regardless of gate output.
    """

    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.local_proj = nn.Linear(feature_dim, hidden_dim)
        self.long_proj = nn.Linear(feature_dim, hidden_dim)
        self.global_proj = nn.Linear(feature_dim, hidden_dim)
        # Gate projects to 3 logits per position; softmax gives mixture weights.
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )
        self.out_proj = nn.Linear(hidden_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        local_context: torch.Tensor,
        long_context: torch.Tensor,
        global_context: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        local_hidden = self.local_proj(local_context)    # (B, T, H)
        long_hidden = self.long_proj(long_context)        # (B, T, H)
        global_hidden = self.global_proj(global_context)  # (B, T, H)

        # Softmax gate: (B, T, 3) weights summing to 1 across the 3 branches.
        gate_logits = self.gate(torch.cat([local_hidden, long_hidden, global_hidden], dim=-1))
        weights = F.softmax(gate_logits, dim=-1)          # (B, T, 3)

        fused = (
            weights[..., 0:1] * local_hidden
            + weights[..., 1:2] * long_hidden
            + weights[..., 2:3] * global_hidden
        )
        return self.norm(residual + self.out_proj(fused))
