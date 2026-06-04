from __future__ import annotations

import torch
import torch.nn as nn


class GatedFeatureFusion(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.local_proj = nn.Linear(feature_dim, hidden_dim)
        self.long_proj = nn.Linear(feature_dim, hidden_dim)
        self.global_proj = nn.Linear(feature_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
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
        local_hidden = self.local_proj(local_context)
        long_hidden = self.long_proj(long_context)
        global_hidden = self.global_proj(global_context)
        gate = self.gate(torch.cat([local_hidden, long_hidden, global_hidden], dim=-1))
        fused = gate * local_hidden + (1.0 - gate) * long_hidden + 0.5 * global_hidden
        return self.norm(residual + self.out_proj(fused))

