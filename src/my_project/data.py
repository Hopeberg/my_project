import torch
from torch.utils.data import Dataset
from torchvision import datasets
from typing import Optional
from torch import Tensor
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


class CustomDataset(Dataset):
    """Custom dataset to load .pt data files.

    Args:
        data_file: Path to the .pt file containing the dataset.
        img_transform: Image transformation to apply.
        target_transform: Target transformation to apply.
    """

    def __init__(
        self,
        data_file: str,
        img_transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
    ) -> None:
        super().__init__()
        self.data = torch.load(data_file)  # Load the .pt file
        self.images = self.data["images"]  # Assumes the images are stored with this key
        self.labels = self.data["labels"]  # Assumes the labels are stored with this key
        self.img_transform = img_transform
        self.target_transform = target_transform

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return image and target tensor."""
        img, target = self.images[idx], self.labels[idx]

        # Apply transformations if specified
        if self.img_transform:
            img = self.img_transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.images)


def download_and_save_data(data_folder: str = "data"):
    """Download the data from torchvision and save it as .pt files."""

    # Define transformation to apply (convert to tensor and)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Download MNIST dataset (apply the transformation during download)
    train_data = datasets.MNIST(root=data_folder, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=data_folder, train=False, download=True, transform=transform)

    # Convert the dataset into tensors
    train_images = torch.stack([img for img, _ in train_data])  # Convert images to a tensor
    train_labels = torch.tensor([label for _, label in train_data])  # Convert labels to a tensor

    test_images = torch.stack([img for img, _ in test_data])  # Convert images to a tensor
    test_labels = torch.tensor([label for _, label in test_data])  # Convert labels to a tensor

    # Save the data as .pt files
    torch.save({"images": train_images, "labels": train_labels}, f"{data_folder}/train_data.pt")
    torch.save({"images": test_images, "labels": test_labels}, f"{data_folder}/test_data.pt")

    print(f"Data saved to {data_folder}/train_data.pt and {data_folder}/test_data.pt")


# Function to get DataLoaders for training, validation, and testing
def get_data_loaders(batch_size: int = 32, data_folder: str = "data"):
    """Returns DataLoaders for training, validation, and testing datasets."""

    # Load .pt data for training and testing
    train_pt_dataset = CustomDataset(data_file=f"{data_folder}/train_data.pt")
    test_pt_dataset = CustomDataset(data_file=f"{data_folder}/test_data.pt")

    # Split the training dataset into training and validation
    train_data, val_data = random_split(
        train_pt_dataset, [int(0.8 * len(train_pt_dataset)), int(0.2 * len(train_pt_dataset))]
    )

    # Create DataLoaders
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_pt_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    # Step 1: Download and save the dataset in .pt format
    download_and_save_data()
