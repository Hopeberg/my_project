import torch
import torch.nn as nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    def __init__(self):
        super(MyAwesomeModel, self).__init__()
        
        # A simple CNN model with one convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # [batch_size, 1, 28, 28] -> [batch_size, 32, 28, 28]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # [batch_size, 32, 28, 28] -> [batch_size, 64, 28, 28]
        self.fc1 = nn.Linear(64 * 28 * 28, 10)  # Flattened input to fully connected layer (10 classes for MNIST)
    
    def forward(self, x):
        # Apply the first convolutional layer
        x = F.relu(self.conv1(x))  # Apply ReLU activation
        
        # Apply the second convolutional layer
        x = F.relu(self.conv2(x))  # Apply ReLU activation
        
        # Flatten the tensor before the fully connected layer
        x = x.view(x.size(0), -1)  # Flatten from [batch_size, 64, 28, 28] to [batch_size, 64*28*28]
        
        # Pass through the fully connected layer to get the final output
        x = self.fc1(x)  # Output shape: [batch_size, 10]
        return x

