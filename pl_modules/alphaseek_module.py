from __future__ import annotations

import torch
import torch.nn as nn

from models.alphaseek_hybrid import AlphaSeekBackbone
from pl_modules.base_module import BaseModule


class AlphaSeekSignalModule(BaseModule):
    def __init__(
        self,
        num_features: int = 6,
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
        action_threshold: float = 0.002,
        transaction_cost: float = 0.0,
        min_action_return: float = 0.0,
        regression_loss_weight: float = 1.0,
        action_loss_weight: float = 0.3,
        lr: float = 0.0005,
        lr_step_size: int = 50,
        lr_gamma: float = 0.5,
        weight_decay: float = 0.0,
        logger_type: str | None = None,
        y_key: str = "Close",
        optimizer: str = "adam",
        mode: str = "default",
        target_mode: str = "price",
        loss: str = "rmse",
        close_index: int = 3,
        **kwargs,
    ):
        super().__init__(
            lr=lr,
            lr_step_size=lr_step_size,
            lr_gamma=lr_gamma,
            weight_decay=weight_decay,
            logger_type=logger_type,
            window_size=window_size,
            y_key=y_key,
            optimizer=optimizer,
            mode=mode,
            loss=loss,
        )
        self.variant = variant
        self.target_mode = target_mode
        self.action_threshold = action_threshold
        self.transaction_cost = transaction_cost
        self.min_action_return = min_action_return
        self.regression_loss_weight = regression_loss_weight
        self.action_loss_weight = action_loss_weight
        self.action_ce = nn.CrossEntropyLoss()

        self.model = AlphaSeekBackbone(
            num_features=num_features,
            window_size=window_size,
            hidden_dim=hidden_dim,
            variant=variant,
            tcn_channels=tcn_channels,
            tcn_kernel_size=tcn_kernel_size,
            tcn_dropout=tcn_dropout,
            num_mamba_branches=num_mamba_branches,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            mamba_use_fast_path=mamba_use_fast_path,
            attention_heads=attention_heads,
            attention_dropout=attention_dropout,
            ffn_hidden_dim=ffn_hidden_dim,
            action_classes=action_classes,
            use_smoothing=use_smoothing,
            close_index=close_index,
        )

    def _price_from_prediction(self, prediction: torch.Tensor, y_old: torch.Tensor) -> torch.Tensor:
        if self.target_mode == "return":
            return y_old.reshape(-1) * (1.0 + prediction.reshape(-1))
        if self.target_mode == "log_return":
            return y_old.reshape(-1) * torch.exp(prediction.reshape(-1))
        if self.target_mode == "delta" or self.mode == "diff":
            return y_old.reshape(-1) + prediction.reshape(-1)
        if self.target_mode == "price":
            return prediction.reshape(-1)
        raise ValueError(f"Unsupported target_mode '{self.target_mode}'")

    def _regression_target(self, y: torch.Tensor, y_old: torch.Tensor) -> torch.Tensor:
        if self.target_mode == "return":
            return (y.reshape(-1) - y_old.reshape(-1)) / y_old.reshape(-1).clamp_min(1e-6)
        if self.target_mode == "log_return":
            return torch.log(y.reshape(-1).clamp_min(1e-6) / y_old.reshape(-1).clamp_min(1e-6))
        if self.target_mode == "delta" or self.mode == "diff":
            return y.reshape(-1) - y_old.reshape(-1)
        if self.target_mode == "price":
            return y.reshape(-1)
        raise ValueError(f"Unsupported target_mode '{self.target_mode}'")

    def forward(self, x, y_old=None):
        prediction = self.model(x).reshape(-1)
        if self.target_mode in {"return", "log_return", "delta"} or self.mode == "diff":
            if y_old is None:
                y_old = x[:, self.model.close_index, -1]
            return self._price_from_prediction(prediction, y_old)
        return prediction

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(x)

    def predict_action_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.predict_action_logits(x)

    def _resolve_regression_loss(self, mse, rmse, mae, mape):
        if self.loss == "mse":
            return mse
        if self.loss == "mae":
            return mae
        if self.loss == "mape":
            return mape
        return rmse

    def _make_action_targets(self, current_price: torch.Tensor, next_price: torch.Tensor) -> torch.Tensor:
        returns = (next_price - current_price) / current_price.clamp_min(1e-6)
        effective_threshold = self.action_threshold + self.min_action_return + self.transaction_cost
        buy = returns > effective_threshold
        sell = returns < -effective_threshold
        labels = torch.full_like(returns, 1, dtype=torch.long)
        labels[buy] = 2
        labels[sell] = 0
        return labels

    def _log_action_distribution(self, action_targets: torch.Tensor, stage: str) -> None:
        counts = torch.bincount(action_targets.detach(), minlength=3).float()
        fractions = counts / counts.sum().clamp_min(1.0)
        for index, name in enumerate(("sell", "hold", "buy")):
            self.log(
                f"{stage}/action_target_{name}_frac",
                fractions[index],
                batch_size=self.batch_size,
                sync_dist=True,
                prog_bar=False,
            )

    def _shared_step(self, batch, stage: str):
        x = batch["features"]
        y = batch[self.y_key]
        y_old = batch[f"{self.y_key}_old"]
        if self.batch_size is None:
            self.batch_size = x.shape[0]

        outputs = self.model(x, return_dict=True)
        prediction = outputs["price"].reshape(-1)
        y_hat = self._price_from_prediction(prediction, y_old)
        action_logits = outputs["action_logits"]

        y_denorm, y_hat_denorm = self.denormalize(y, y_hat)
        y_old_denorm = self.denormalize_value(y_old)

        mse = self.mse(y_hat_denorm, y_denorm)
        rmse = torch.sqrt(mse)
        mape = self.mape(y_hat_denorm, y_denorm)
        mae = self.l1(y_hat_denorm, y_denorm)

        action_targets = self._make_action_targets(y_old_denorm.reshape(-1), y_denorm.reshape(-1))
        action_loss = self.action_ce(action_logits, action_targets)
        action_acc = (action_logits.argmax(dim=-1) == action_targets).float().mean()
        self._log_action_distribution(action_targets, stage)

        regression_target = self._regression_target(y_denorm, y_old_denorm)
        prediction_target = prediction if self.normalization_coeffs is None else self._regression_target(y_hat_denorm, y_old_denorm)
        regression_mse = self.mse(prediction_target, regression_target)
        regression_rmse = torch.sqrt(regression_mse)
        regression_mae = self.l1(prediction_target, regression_target)
        regression_mape = self.mape(prediction_target, regression_target) if self.target_mode == "price" else regression_mae
        regression_loss = self._resolve_regression_loss(regression_mse, regression_rmse, regression_mae, regression_mape)
        total_loss = self.regression_loss_weight * regression_loss + self.action_loss_weight * action_loss

        self.log(f"{stage}/loss", total_loss.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=(stage != "test"))
        self.log(f"{stage}/mse", mse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        self.log(f"{stage}/rmse", rmse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log(f"{stage}/mape", mape.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        self.log(f"{stage}/mae", mae.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        self.log(f"{stage}/action_loss", action_loss.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        self.log(f"{stage}/action_acc", action_acc.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, stage="val")
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, stage="test")
        return {"test_loss": loss}

