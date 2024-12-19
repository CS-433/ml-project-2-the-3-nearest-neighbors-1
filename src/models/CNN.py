import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderDecoderCNN(nn.Module):
    def __init__(self):
        super(EncoderDecoderCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Input: 400x400x3, Output: 400x400x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) # Output: 400x400x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)                 # Output: 200x200x32

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # Output: 200x200x64
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # Output: 200x200x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)                 # Output: 100x100x128

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # Output: 200x200x64
        self.conv5 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1) # Output: 200x200x32

        self.upconv2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2) # Output: 400x400x16
        self.conv6 = nn.Conv2d(16, 1, kernel_size=1, stride=1)             # Output: 400x400x1

    def forward(self, x):
        # Encoder
        x = F.relu(self.conv1(x))                                         # Output: 400x400x16
        x = F.relu(self.conv2(x))                                         # Output: 400x400x32
        x = self.pool1(x)                                                 # Output: 200x200x32

        x = F.relu(self.conv3(x))                                         # Output: 200x200x64
        x = F.relu(self.conv4(x))                                         # Output: 200x200x128
        x = self.pool2(x)                                                 # Output: 100x100x128

        # Decoder
        x = F.relu(self.upconv1(x))                                       # Output: 200x200x64
        x = F.relu(self.conv5(x))                                         # Output: 200x200x32

        x = F.relu(self.upconv2(x))                                       # Output: 400x400x16
        x = torch.sigmoid(self.conv6(x))                                  # Output: 400x400x1
        return x