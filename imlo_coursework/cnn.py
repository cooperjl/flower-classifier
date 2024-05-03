import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        self.batch_norm1 = nn.BatchNorm2d(num_features=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.batch_norm2 = nn.BatchNorm2d(num_features=32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.batch_norm3 = nn.BatchNorm2d(num_features=64)
        # pooling layer(s)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # fully connected layer(s)
        self.fc1 = nn.Linear(in_features=12544, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=102)
        # dropout regularisation
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.batch_norm1(self.conv1(x))
        x = self.pool(F.relu(x))
        x = self.drop(x)
        x = self.batch_norm2(self.conv2(x))
        x = self.pool(F.relu(x))
        x = self.drop(x)
        x = self.batch_norm3(self.conv3(x))
        x = self.pool(F.relu(x))
        x = self.drop(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop(self.fc2(x))

        return x
