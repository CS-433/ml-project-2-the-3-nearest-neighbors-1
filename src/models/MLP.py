import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, img_size=400):
        super().__init__()


        input_size = img_size*img_size*3
        output_size = img_size*img_size

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size,500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,500),
            nn.ReLU(),
            nn.Linear(500,output_size),
            nn.Sigmoid(),
            nn.Unflatten(dim=1, unflattened_size=(1, img_size, img_size)),
        )

    def forward(self, x):
        return self.layers(x)