import matplotlib.pyplot as plt
import torch
import typer
from data import CustomDataset  
import matplotlib.pyplot as plt
import torch

def show_image_and_target(images, targets, show=True):
    """
    Display images and their corresponding target labels.

    Args:
        images (torch.Tensor): A tensor of images (shape: [num_images, 1, height, width]).
        targets (torch.Tensor): A tensor of labels (shape: [num_images]).
        show (bool): If True, show the plot; otherwise, save the plot.
    """
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))

    # Flatten the axes to easily iterate over
    axes = axes.flatten()

    # Loop through the first 25 images and labels
    for i in range(25):
        ax = axes[i]
        image = images[i]
        label = targets[i]

        # Normalize the image to the range [0, 1] and then scale to [0, 255]
        image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
        image = image * 255  # Scale to [0, 255]

        # Convert the image to uint8 for displaying
        image = image.to(torch.uint8)

        # Display the image
        ax.imshow(image.squeeze(), cmap="gray")  # Squeeze removes the single channel dimension
        ax.set_title(f"Label: {label.item()}")  # Set the title with the label
        ax.axis("off")  # Turn off axis

    # Adjust layout to make room for titles and remove extra space
    plt.tight_layout()

    # If `show` is True, display the plot, otherwise save the plot
    if show:
        plt.show()
    else:
        plt.close()


def dataset_statistics(datadir: str = "data") -> None:
    """Compute dataset statistics."""
    # Load training and test datasets using the CustomDataset class
    # Ensure correct paths to the .pt files
    train_dataset = CustomDataset(data_file=f"{datadir}/train_data.pt")
    test_dataset = CustomDataset(data_file=f"{datadir}/test_data.pt")
    
    # Print basic statistics about the training dataset
    print(f"Train dataset: {train_dataset.__class__.__name__}")
    print(f"Number of images: {len(train_dataset)}")
    print(f"Image shape: {train_dataset[0][0].shape}")  # Assuming each data point is a tuple (image, label)
    print("\n")

    # Print basic statistics about the test dataset
    print(f"Test dataset: {test_dataset.__class__.__name__}")
    print(f"Number of images: {len(test_dataset)}")
    print(f"Image shape: {test_dataset[0][0].shape}")  # Assuming each data point is a tuple (image, label)

    # Visualize and save first 25 images in the training set (if show_image_and_target is implemented)
    show_image_and_target(train_dataset.images[:25], train_dataset.labels[:25], show=False)
    plt.savefig("mnist_images.png")
    plt.close()

    # Plot and save the label distribution for the training set
    train_label_distribution = torch.bincount(train_dataset.labels)
    plt.bar(torch.arange(10), train_label_distribution)
    plt.title("Train label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("train_label_distribution.png")
    plt.close()

    # Plot and save the label distribution for the test set
    test_label_distribution = torch.bincount(test_dataset.labels)
    plt.bar(torch.arange(10), test_label_distribution)
    plt.title("Test label distribution")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig("test_label_distribution.png")
    plt.close()

if __name__ == "__main__":
    typer.run(dataset_statistics)