import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        # pooling layer(s)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # fully connected layer(s)
        self.fc1 = nn.Linear(in_features=115200, out_features=128) # 39200 # 115200
        self.fc2 = nn.Linear(in_features=128, out_features=classes)

    def forward(self, x):
        # go through conv layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # go through fc layers
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
