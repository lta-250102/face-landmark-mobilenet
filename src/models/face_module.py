from typing import Any, Tuple
from torchmetrics import MinMetric, MeanMetric
from albumentations.pytorch.transforms import ToTensorV2
import torch
from lightning import LightningModule
from PIL import Image
import albumentations as A
import numpy as np


class FaceModule(LightningModule):
    def __init__(self, n_classes: int = 136, lr: float = 1e-4) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
        self.net.classifier = torch.nn.Linear(1280, self.hparams.n_classes)
        self.criterion = torch.nn.MSELoss()

         # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def on_train_start(self) -> None:
        self.train_loss.reset()
        self.val_loss.reset()
        self.val_loss_best.reset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
    
    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.train_loss(loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.val_loss(loss)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.test_loss(loss)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def predict_step(self, img: Image) -> Any:
        t = A.Compose([
            A.Resize(244, 244),
            A.Normalize(),
            ToTensorV2(),
        ])
        transformed = t(image=np.array(img))
        image = transformed['image']
        logits = self.forward(image.unsqueeze(0))
        return logits.squeeze(0)
    
    def on_validation_epoch_end(self) -> None:
        acc = self.val_loss.compute()  # get current val loss
        self.val_loss_best(acc)  # update best so far val loss
        self.log("val_loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def save_to_state_dict(self, path: str):
        torch.save(self.state_dict(), path)

    def load_from_state_dict(self, path: str):
        self.load_state_dict(torch.load(path))
     

if __name__ == "__main__":
    _ = FaceModule(None)
