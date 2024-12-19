import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
import torch.nn.functional as F

class RoadBackgroundClassifierCNN(nn.Module):
    def __init__(self):
        super(RoadBackgroundClassifierCNN, self).__init__()
        
        # Encoder: Convolutional and pooling layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)  # Output: 16x16x16
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1) # Output: 32x16x16
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                                          # Output: 32x8x8
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1) # Output: 64x8x8
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # Output: 128x8x8
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                                          # Output: 128x4x4
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 64)           # Second hidden layer
        self.fc3 = nn.Linear(64, 1)             # Output layer for binary classification

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))  # Conv1 + ReLU
        x = F.relu(self.conv2(x))  # Conv2 + ReLU
        x = self.pool1(x)          # Pooling 1

        x = F.relu(self.conv3(x))  # Conv3 + ReLU
        x = F.relu(self.conv4(x))  # Conv4 + ReLU
        x = self.pool2(x)          # Pooling 2

        # Flatten the output for the fully connected layers
        x = x.view(-1, 128 * 4 * 4)  
        x = F.relu(self.fc1(x))  # Fully connected layer 1
        x = F.relu(self.fc2(x))  # Fully connected layer 2
        x = torch.sigmoid(self.fc3(x))  # Sigmoid for binary classification
        return x