from __future__ import annotations

import torch
import torch.nn as nn

from models.cmamba import Mamba
from models.layers.fusion import GatedFeatureFusion
from models.layers.smoothing import AdaptiveSignalSmoother
from models.layers.tcn import TemporalConvNet


class InputPreprocessor(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)
        self.layer_norm = nn.LayerNorm(num_features)
        self.proj = nn.Linear(num_features, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.batch_norm(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        x = self.activation(self.proj(x))
        return self.dropout(x)


class ParallelMambaEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_branches: int,
        d_state: int,
        d_conv: int,
        expand: int,
        dropout: float,
        use_fast_path: bool = False,
    ):
        super().__init__()
        if hidden_dim % num_branches != 0:
            raise ValueError("hidden_dim must be divisible by num_branches")
        branch_dim = hidden_dim // num_branches
        self.pre_norm = nn.LayerNorm(hidden_dim)
        self.pre_proj = nn.Linear(hidden_dim, hidden_dim)
        self.branches = nn.ModuleList(
            [
                Mamba(
                    d_model=branch_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    use_fast_path=use_fast_path,
                )
                for _ in range(num_branches)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.pre_proj(self.pre_norm(x))
        chunks = base.chunk(len(self.branches), dim=-1)
        outputs = [branch(chunk) for branch, chunk in zip(self.branches, chunks)]
        combined = torch.cat(outputs, dim=-1)
        return self.out_norm(x + self.dropout(combined))


class AlphaSeekBackbone(nn.Module):
    def __init__(
        self,
        num_features: int,
        window_size: int = 30,
        hidden_dim: int = 64,
        variant: str = "hybrid",
        tcn_channels: list[int] | None = None,
        tcn_kernel_size: int = 3,
        tcn_dropout: float = 0.1,
        num_mamba_branches: int = 4,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        mamba_use_fast_path: bool = False,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        ffn_hidden_dim: int = 128,
        action_classes: int = 3,
        use_smoothing: bool = True,
        close_index: int = 3,
    ):
        super().__init__()
        self.window_size = window_size
        self.variant = variant
        self.close_index = close_index

        tcn_channels = list(tcn_channels or [hidden_dim, hidden_dim])
        if tcn_channels[-1] != hidden_dim:
            tcn_channels.append(hidden_dim)

        self.preprocessor = InputPreprocessor(num_features=num_features, hidden_dim=hidden_dim, dropout=tcn_dropout)
        self.tcn_branch = TemporalConvNet(hidden_dim, tcn_channels, kernel_size=tcn_kernel_size, dropout=tcn_dropout)
        self.mamba_branch = ParallelMambaEncoder(
            hidden_dim=hidden_dim,
            num_branches=num_mamba_branches,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
            dropout=tcn_dropout,
            use_fast_path=mamba_use_fast_path,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.fusion = GatedFeatureFusion(feature_dim=hidden_dim, hidden_dim=hidden_dim)
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(attention_dropout),
            nn.Linear(ffn_hidden_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.smoother = AdaptiveSignalSmoother(hidden_dim=hidden_dim) if use_smoothing else None
        self.regression_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.action_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, action_classes),
        )

    def _compose_sequence(
        self,
        base: torch.Tensor,
        local_context: torch.Tensor,
        long_context: torch.Tensor,
        global_context: torch.Tensor,
    ) -> torch.Tensor:
        if self.variant == "tcn":
            return self.output_norm(base + local_context + self.ffn(local_context))
        if self.variant == "mamba":
            return self.output_norm(base + long_context + self.ffn(long_context))
        if self.variant == "attention":
            return self.output_norm(base + global_context + self.ffn(global_context))
        fused = self.fusion(local_context, long_context, global_context, residual=base)
        return self.output_norm(fused + self.ffn(fused))

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        base = self.preprocessor(x)
        local_context = self.tcn_branch(base)
        long_context = self.mamba_branch(base)
        global_context, attn_weights = self.attention(
            self.attention_norm(base),
            self.attention_norm(base),
            self.attention_norm(base),
            need_weights=True,
        )
        sequence = self._compose_sequence(base, local_context, long_context, global_context)
        if self.smoother is not None:
            sequence = self.smoother(sequence, raw_inputs=x, close_index=self.close_index)
        pooled = sequence[:, -1, :]
        return {
            "sequence": sequence,
            "pooled": pooled,
            "attention_weights": attn_weights,
        }

    def forward_heads(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pooled = features["pooled"]
        return {
            "price": self.regression_head(pooled),
            "action_logits": self.action_head(pooled),
        }

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)["pooled"]

    def predict_action_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x, return_dict=True)["action_logits"]

    def forward(self, x: torch.Tensor, return_dict: bool = False):
        features = self.forward_features(x)
        heads = self.forward_heads(features)
        output = {**features, **heads}
        if return_dict:
            return output
        return output["price"]

