import torch
import torch.nn as nn
import torch.nn.functional as F

class RoadSegmentationCNN(nn.Module):

    def _init_(self):
        super()._init_()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces dimensions by half

        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # Upsample from 200x200 to 400x400
        self.upconv2 = nn.Conv2d(16, 1, kernel_size=1, stride=1)  # Reduce to 1 channel

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))  # Output: [batch_size, 16, 200, 200]
        x = self.pool(torch.relu(self.conv2(x)))  # Output: [batch_size, 32, 100, 100]

        x = torch.relu(self.upconv1(x))  # Upsample to [batch_size, 16, 200, 200]
        x = torch.sigmoid(self.upconv2(x))  # Final output: [batch_size, 1, 400, 400]
        return x