import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        
        input_size = 400*400*3
        output_size = 400*400

        self.fc1 = nn.Linear(input_size,200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,200)
        self.fc4 = nn.Linear(200,output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        preds = F.sigmoid(self.fc4(x))
        return preds
