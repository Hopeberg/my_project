import os
import torch
from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def normalize(images: torch.Tensor) -> torch.Tensor:
    """Normalize images."""
    return (images - images.mean()) / images.std()


def download_and_save_fashion_mnist(raw_dir: str = "data/raw") -> None:
    """Download and save FashionMNIST data to the raw directory."""
    # Resolve the full path relative to the project root (go up one directory from src)
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the script
    project_root_dir = os.path.abspath(os.path.join(base_dir, "../.."))  # Go up two directories (src -> my_project)
    full_raw_dir = os.path.join(project_root_dir, raw_dir)  # Combine with raw_dir

    # Ensure the raw directory exists
    os.makedirs(full_raw_dir, exist_ok=True)
    print(f"Saving raw data to: {full_raw_dir}")

    # Download and load the training data
    trainset = datasets.FashionMNIST(
        root=os.path.expanduser("~/.pytorch/F_MNIST_data/"), download=True, train=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False)

    # Download and load the test data
    testset = datasets.FashionMNIST(
        root=os.path.expanduser("~/.pytorch/F_MNIST_data/"), download=True, train=False, transform=transform
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False)

    # Save training data in chunks
    for train_images, train_targets in trainloader:
        for i in range(6):  # Split into 6 chunks
            chunk_size = len(train_images) // 6
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size if i < 5 else len(train_images)
            os.makedirs(full_raw_dir, exist_ok=True)  # Ensure the directory exists before saving
            torch.save(train_images[start_idx:end_idx], os.path.join(full_raw_dir, f"train_images_{i}.pt"))
            torch.save(train_targets[start_idx:end_idx], os.path.join(full_raw_dir, f"train_target_{i}.pt"))
            print(f"Saved train chunk {i} to: {full_raw_dir}")

    # Save test data
    for test_images, test_targets in testloader:
        os.makedirs(full_raw_dir, exist_ok=True)  # Ensure the directory exists before saving
        torch.save(test_images, os.path.join(full_raw_dir, "test_images.pt"))
        torch.save(test_targets, os.path.join(full_raw_dir, "test_target.pt"))
        print(f"Saved test data to: {full_raw_dir}")

    print(f"FashionMNIST data has been downloaded and saved to {full_raw_dir}")


def preprocess_data(raw_dir: str = "data/raw", processed_dir: str = "data/processed") -> None:
    """Process raw data and save it to processed directory."""
    # Resolve the full path for raw_dir
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the script
    project_root_dir = os.path.abspath(os.path.join(base_dir, "../.."))  # Go up two directories (src -> my_project)
    full_raw_dir = os.path.join(project_root_dir, raw_dir)
    full_processed_dir = os.path.join(project_root_dir, processed_dir)

    os.makedirs(full_processed_dir, exist_ok=True)
    print(f"Saving processed data to: {full_processed_dir}")

    train_images, train_target = [], []
    for i in range(6):
        train_images.append(torch.load(f"{full_raw_dir}/train_images_{i}.pt"))
        train_target.append(torch.load(f"{full_raw_dir}/train_target_{i}.pt"))
    train_images = torch.cat(train_images)
    train_target = torch.cat(train_target)

    test_images: torch.Tensor = torch.load(f"{full_raw_dir}/test_images.pt")
    test_target: torch.Tensor = torch.load(f"{full_raw_dir}/test_target.pt")

    # Normalize the data
    train_images = train_images.unsqueeze(1).float()
    test_images = test_images.unsqueeze(1).float()
    train_target = train_target.long()
    test_target = test_target.long()

    train_images = normalize(train_images)
    test_images = normalize(test_images)

    # Save processed data
    torch.save(train_images, f"{full_processed_dir}/train_images.pt")
    torch.save(train_target, f"{full_processed_dir}/train_target.pt")
    torch.save(test_images, f"{full_processed_dir}/test_images.pt")
    torch.save(test_target, f"{full_processed_dir}/test_target.pt")
    print(f"Processed data has been saved to {full_processed_dir}")


def corrupt_mnist() -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Return train and test datasets for corrupt MNIST."""
    # Load the processed data
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the script
    project_root_dir = os.path.abspath(os.path.join(base_dir, "../.."))  # Go up two directories (src -> my_project)
    processed_dir = os.path.join(project_root_dir, "data/processed")

    train_images = torch.load(os.path.join(processed_dir, "train_images.pt"))
    train_target = torch.load(os.path.join(processed_dir, "train_target.pt"))
    test_images = torch.load(os.path.join(processed_dir, "test_images.pt"))
    test_target = torch.load(os.path.join(processed_dir, "test_target.pt"))

    train_set = torch.utils.data.TensorDataset(train_images, train_target)
    test_set = torch.utils.data.TensorDataset(test_images, test_target)
    return train_set, test_set


if __name__ == "__main__":
    download_and_save_fashion_mnist()  # Download data and save it to raw
    preprocess_data()  # Preprocess and save data to processed
