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

        self.fc1 = nn.Linear(input_size,200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,200)
        self.fc4 = nn.Linear(200,output_size)

    def forward(self, x):
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        preds = F.sigmoid(self.fc4(x))
        return preds
