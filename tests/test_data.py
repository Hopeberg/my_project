from torchvision.datasets import MNIST
from torchvision import transforms
import pytest
import torch
from torchvision.datasets import MNIST
from torchvision import transforms
import torch
import pytest
import os
import os.path


# @pytest.mark.skipif(not os.path.exists( _PATH_DATA), reason="Data files not found")
_PATH_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))


def test_training_data_size():
    print(f"Path exists: {os.path.exists(_PATH_DATA)}, Path: {_PATH_DATA}")
    print(f"Contents of path: {os.listdir(_PATH_DATA) if os.path.exists(_PATH_DATA) else 'Path does not exist'}")
    # Initialize the training dataset
    dataset = MNIST(root=_PATH_DATA, train=True, download=True, transform=transforms.ToTensor())

    # Check the number of training examples
    assert len(dataset) == 60000, "Training dataset size mismatch."


def test_training_data_shape():
    # Initialize the training dataset
    dataset = MNIST(root=_PATH_DATA, train=True, download=True, transform=transforms.ToTensor())

    # Check the shape of each datapoint (checking for 10 random datapoints)
    for i in range(10):
        img, label = dataset[i]
        assert img.shape == (1, 28, 28), f"Data point {i} shape mismatch."


def test_training_data_labels():
    # Initialize the training dataset
    dataset = MNIST(root=_PATH_DATA, train=True, download=True, transform=transforms.ToTensor())

    # Verify that all labels are represented
    labels = [dataset[i][1] for i in range(len(dataset))]
    unique_labels = set(labels)
    assert unique_labels == set(range(10)), "Not all labels are represented in the training set."


def test_test_data_size():
    # Initialize the test dataset
    test_dataset = MNIST(root=_PATH_DATA, train=False, download=True, transform=transforms.ToTensor())

    # Check the number of test examples
    assert len(test_dataset) == 10000, "Test dataset size mismatch."


def test_test_data_shape():
    # Initialize the test dataset
    test_dataset = MNIST(root=_PATH_DATA, train=False, download=True, transform=transforms.ToTensor())

    # Check the shape of each datapoint (checking for 10 random datapoints)
    for i in range(10):
        img, label = test_dataset[i]
        assert img.shape == (1, 28, 28), f"Test data point {i} shape mismatch."


def test_test_data_labels():
    # Initialize the test dataset
    test_dataset = MNIST(root=_PATH_DATA, train=False, download=True, transform=transforms.ToTensor())

    # Verify that all labels are represented
    labels = [test_dataset[i][1] for i in range(len(test_dataset))]
    unique_labels = set(labels)
    assert unique_labels == set(range(10)), "Not all labels are represented in the test set."
