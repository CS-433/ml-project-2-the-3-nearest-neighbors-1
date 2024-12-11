import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, size_image=400):
        super().__init__()
        
        
        input_size = size_image*size_image*3
        output_size = size_image*size_image
        
        self.layers = torch.nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size,500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,output_size),
            # nn.Sigmoid(),
            nn.Unflatten(dim=1, unflattened_size=(1, size_image, size_image)),  
        )

    def forward(self, x):
        return self.layers(x)


