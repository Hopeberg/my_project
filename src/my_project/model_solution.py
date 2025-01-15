import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl


# MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_set = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_set = datasets.MNIST(root="data", train=False, download=True, transform=transform)

# Split training set into training and validation subsets
train_data, val_data = random_split(train_set, [55000, 5000])

# Create DataLoaders
train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=32)
test_dataloader = DataLoader(test_set, batch_size=32)


# LightningModule definition
class MyAwesomeModel(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if x.ndim != 4:
            raise ValueError("Expected input to a 4D tensor")
        if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError("Expected each sample to have shape [1, 28, 28]")
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)

    def training_step(self, batch) -> None:
        data, target = batch
        preds = self(data)
        loss = self.loss_fn(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        # self.log('train_loss', loss, on_epoch=True)
        # self.log('train_acc', acc, on_epoch=True)
        # self.logger.experiment.log({'logits': wandb.Histogram(preds.detach().cpu())})
        return loss

    def validation_step(self, batch) -> None:
        data, target = batch
        preds = self(data)
        loss = self.loss_fn(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)

    def test_step(self, batch, batch_idx) -> None:
        data, target = batch
        preds = self(data)
        loss = self.loss_fn(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


if __name__ == "__main__":
    model = MyAwesomeModel()
    checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")
    trainer = Trainer(
        max_epochs=10,
        limit_train_batches=0.2,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=pl.loggers.WandbLogger(project="dtu_mlops"),
    )
    trainer.fit(model, train_dataloader, val_dataloader)
    trainer.test(model, test_dataloader)

    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    # Testing with dummy input
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
