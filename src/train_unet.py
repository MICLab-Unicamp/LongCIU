'''
Trains a baseline 2D UNet on the LongCIU split.
'''
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Any
from monai.networks.nets.unet import UNet
from monai.losses.dice import DiceLoss
from longciu import LongCIUDataModule, TrainTransform


class LongCIUNet(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = UNet(spatial_dims=2, 
                          in_channels=1, 
                          out_channels=3, 
                          channels=(32, 64, 128, 256, 512),
                          strides=(2, 2, 2, 2))
        self.loss = DiceLoss(include_background=True, to_onehot_y=True, softmax=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, train_batch, _):
        x, y, _ = train_batch
        y_hat = self.model(x)

        train_loss = self.loss(y_hat, y)
        self.log("train_loss", train_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.hparams.train_batch_size)

        return train_loss

    def validation_step(self, val_batch, _):
        x, y, _ = val_batch
        y_hat = self.model(x)

        val_loss = self.loss(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=self.hparams.eval_batch_size)
    
    def configure_optimizers(self):
        return torch.optim.Adam(params=self.parameters(), lr=1e-4)


if __name__ == "__main__":
    import argparse
    import torchinfo

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="baseline_unet")
    parser.add_argument("--data_dir", type=str, default=os.path.join("..", "data"))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=30)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    hparams = parser.parse_args()
    
    if torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    if hparams.debug:
        callbacks = None
        logger = None
    else:
        callbacks = [ModelCheckpoint(dirpath="logs",
                                     filename=hparams.name + "_{epoch}-{val_loss:.2f}",
                                     monitor="val_loss",
                                     mode="min")]
        logger = TensorBoardLogger(save_dir="logs",
                                   name=hparams.name)

    trainer = pl.Trainer(max_epochs=5000,
                         accelerator=accelerator,
                         devices=1 if accelerator == "cpu" else hparams.ngpus,
                         precision=32,
                         logger=logger,
                         callbacks=callbacks,
                         fast_dev_run=hparams.debug,
                         enable_checkpointing=False if hparams.debug else True)
    
    model = LongCIUNet(hparams)
    torchinfo.summary(model, input_size=(1, 1, 256, 256))
    data_module = LongCIUDataModule(data_dir=hparams.data_dir,
                                    num_workers=hparams.num_workers,
                                    train_batch_size=hparams.train_batch_size,
                                    eval_batch_size=hparams.eval_batch_size,
                                    train_transform=TrainTransform(),
                                    eval_transform=None)

    trainer.fit(model, data_module)
