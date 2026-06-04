import torch
import torch.nn as nn
from torchmetrics.regression import MeanAbsolutePercentageError as MAPE

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl
    

class BaseModule(pl.LightningModule):

    def __init__(
        self,
        lr=0.0002, 
        lr_step_size=50,
        lr_gamma=0.1,
        weight_decay=0.0, 
        logger_type=None,
        window_size=14,
        y_key='Close',
        optimizer='adam',
        mode='default',
        loss='rmse',
        lr_scheduler: str = 'step',
        lr_cosine_T_max: int = 100,
        lr_cosine_eta_min: float = 1e-6,
    ):
        super().__init__()

        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.logger_type = logger_type 
        self.y_key = y_key
        self.optimizer = optimizer
        self.batch_size = None   
        self.mode = mode
        self.window_size = window_size
        self.loss = loss
        self.lr_scheduler = lr_scheduler
        self.lr_cosine_T_max = lr_cosine_T_max
        self.lr_cosine_eta_min = lr_cosine_eta_min

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.mape = MAPE()
        self.normalization_coeffs = None

    def forward(self, x, y_old=None):
        if self.mode == 'default':
            return self.model(x).reshape(-1)
        elif self.mode == 'diff':
            return self.model(x).reshape(-1) + y_old
        
    def set_normalization_coeffs(self, factors):
        if factors is None:
            return
        scale = factors.get(self.y_key).get('max') - factors.get(self.y_key).get('min')
        shift = factors.get(self.y_key).get('min')
        self.normalization_coeffs = (scale, shift)

    def denormalize(self, y, y_hat):
        if self.normalization_coeffs is not None:
            scale, shift = self.normalization_coeffs
            y = y * scale + shift
            y_hat = y_hat * scale + shift
        return y, y_hat

    def denormalize_value(self, value):
        if self.normalization_coeffs is None:
            return value
        scale, shift = self.normalization_coeffs
        return value * scale + shift

    def training_step(self, batch, batch_idx):
        x = batch['features']
        y = batch[self.y_key]
        y_old = batch[f'{self.y_key}_old']
        if self.batch_size is None:
            self.batch_size = x.shape[0]
        y_hat = self.forward(x, y_old).reshape(-1)
        y, y_hat = self.denormalize(y, y_hat)
        mse = self.mse(y_hat, y)
        rmse = torch.sqrt(mse)
        mape = self.mape(y_hat, y)
        l1 = self.l1(y_hat, y)

        self.log("train/mse", mse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        self.log("train/rmse", rmse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("train/mape", mape.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("train/mae", l1.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)

        if self.loss == 'mse':
            return mse
        elif self.loss == 'rmse':
            return rmse
        elif self.loss == 'mae':
            return l1
        elif self.loss == 'mape':
            return mape
        
    def validation_step(self, batch, batch_idx):
        x = batch['features']
        y = batch[self.y_key]
        y_old = batch[f'{self.y_key}_old']
        if self.batch_size is None:
            self.batch_size = x.shape[0]
        y_hat = self.forward(x, y_old).reshape(-1)
        y, y_hat = self.denormalize(y, y_hat)
        mse = self.mse(y_hat, y)
        rmse = torch.sqrt(mse)
        mape = self.mape(y_hat, y)
        l1 = self.l1(y_hat, y)

        self.log("val/mse", mse.detach(), sync_dist=True, batch_size=self.batch_size, prog_bar=False)
        self.log("val/rmse", rmse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("val/mape", mape.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("val/mae", l1.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        return {"val_loss": mse}
    
    def test_step(self, batch, batch_idx):
        x = batch['features']
        y = batch[self.y_key]
        y_old = batch[f'{self.y_key}_old']
        if self.batch_size is None:
            self.batch_size = x.shape[0]
        y_hat = self.forward(x, y_old).reshape(-1)
        y, y_hat = self.denormalize(y, y_hat)
        mse = self.mse(y_hat, y)
        rmse = torch.sqrt(mse)
        mape = self.mape(y_hat, y)
        l1 = self.l1(y_hat, y)

        self.log("test/mse", mse.detach(), sync_dist=True, batch_size=self.batch_size, prog_bar=False)
        self.log("test/rmse", rmse.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("test/mape", mape.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        self.log("test/mae", l1.detach(), batch_size=self.batch_size, sync_dist=True, prog_bar=False)
        return {"test_loss": mse}
    
    def configure_optimizers(self):
        if self.optimizer == 'adam':
            optim = torch.optim.Adam(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == 'adamw':
            # AdamW decouples weight decay from the gradient update — better
            # for models with attention / LayerNorm (like this one).
            optim = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        elif self.optimizer == 'sgd':
            optim = torch.optim.SGD(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f'Unimplemented optimizer {self.optimizer}')

        if self.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim,
                T_max=self.lr_cosine_T_max,
                eta_min=self.lr_cosine_eta_min,
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optim, self.lr_step_size, self.lr_gamma
            )

        return [optim], [scheduler]

    def lr_scheduler_step(self, scheduler, *args, **kwargs):
        scheduler.step()
