from src.my_project.model_solution import MyAwesomeModel
import torch
import pytest
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import random_split
from pytorch_lightning.loggers import WandbLogger
import os
import platform

if platform.system() == "Darwin":  # macOS
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # Disable allocation limits
    torch.backends.mps.is_available = lambda: False  # Disable MPS backend
    torch.backends.mps.is_built = lambda: False  # Ensure MPS is not used


def test_model_initialization():
    model = MyAwesomeModel()

    # Check the model architecture
    assert isinstance(model, MyAwesomeModel), "Model is not of the correct type"

    # Ensure that the model has the correct number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params > 0, "Model has zero parameters"


def test_configure_optimizers():
    model = MyAwesomeModel()

    # Get the optimizer from the model's configure_optimizers method
    optimizers = model.configure_optimizers()

    # Check that we get an Adam optimizer
    assert isinstance(optimizers, torch.optim.Optimizer), "Optimizer is not an instance of torch.optim.Optimizer"
    assert isinstance(optimizers.param_groups[0]["params"], list), "Optimizer parameters are not a list"


def test_trainer():
    # MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_set = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    # Split training set into training and validation subsets
    train_data, val_data = random_split(train_set, [55000, 5000])

    # Create DataLoaders
    train_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True)
    val_dataloader = DataLoader(val_data, batch_size=16, num_workers=4, persistent_workers=True)
    test_dataloader = DataLoader(test_set, batch_size=16, num_workers=4, persistent_workers=True)

    model = MyAwesomeModel()

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, verbose=True, mode="min")

    # Initialize a logger
    logger = WandbLogger(project="dtu_mlops")

    # Initialize Trainer
    trainer = Trainer(
        max_epochs=1,  # Run for 1 epoch for testing
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=logger,
    )

    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)
    assert trainer.current_epoch == 1, "Trainer did not train for 1 epoch"

    # Test the model (using the test set)
    test_results = trainer.test(model, test_dataloader)
    assert test_results is not None, "Testing phase failed"


def test_model_output_shape():
    model = MyAwesomeModel()

    # Create a dummy input tensor of the correct shape
    dummy_input = torch.randn(1, 1, 28, 28)  # 1 sample, 1 channel, 28x28 image

    # Get model output
    output = model(dummy_input)

    # Check the output shape (should be (1, 10) for classification into 10 classes)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), but got {output.shape}"
