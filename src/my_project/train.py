import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import typer
from torch.utils.data import DataLoader
from my_project.model import MyAwesomeModel  # Import your custom model
from my_project.data import corrupt_mnist  # Import the custom dataset loading function
import os

# Define the device to run the model (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10, model_path: str = "models/model.pth", plot_path: str = "reports/figures/training_statistics.png") -> None:
    """Train a model on MNIST."""
    print(f"Training with {lr=}, {batch_size=}, {epochs=}")
    print(f"Using device: {DEVICE}")
    
    # Load the MNIST dataset using the custom `corrupt_mnist` function
    train_set, _ = corrupt_mnist()  # Assuming corrupt_mnist returns (train_set, test_set)
    
    # Create the DataLoader for training data
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    # Initialize the model, loss function, and optimizer
    model = MyAwesomeModel().to(DEVICE)  # Move the model to the correct device
    loss_fn = nn.CrossEntropyLoss()  # Loss function for classification
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Adam optimizer
    
    # Track statistics for loss and accuracy
    statistics = {"train_loss": [], "train_accuracy": []}
    
    for epoch in range(epochs):
        model.train()  # Set the model to training mode
        for i, (img, target) in enumerate(train_dataloader):
            print(f"Input shape: {img.shape}")  # Debugging the input shape
            
            # Ensure the image tensor has the correct shape
            if img.dim() != 4:  # Check for the correct dimensions [batch_size, channels, height, width]
                raise ValueError(f"Input tensor has incorrect shape: {img.shape}")

            # If the shape is still wrong, we need to reshape or fix the data loading
            if img.shape[2] == 1:  # Check if the shape is [batch_size, channels, 1, 28, 28]
                img = img.squeeze(2)  # Remove the extra dimension

            print(f"After squeeze: {img.shape}")  # Check the shape after squeezing
            
            img, target = img.to(DEVICE), target.to(DEVICE)  # Move data to the device
            
            optimizer.zero_grad()  # Zero the gradients before backpropagation
            y_pred = model(img)  # Forward pass
            
            # Make sure the target is the right shape (it should be 1D: [batch_size])
            target = target.view(-1)  # Flatten target to [batch_size]
            
            loss = loss_fn(y_pred, target)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update the model parameters
            
            # Track loss and accuracy
            statistics["train_loss"].append(loss.item())
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)
            
            if i % 100 == 0:  # Print every 100 iterations
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch} complete")
    
    # Ensure the 'models' directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the trained model to disk
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Plot and save the training statistics
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)  # Ensure the 'reports/figures' folder exists
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(plot_path)
    print(f"Training statistics saved to {plot_path}")

# Add a main entry point for Typer
def main():
    typer.run(train)

if __name__ == "__main__":
    main()











