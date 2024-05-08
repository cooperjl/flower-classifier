import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)

        # batch norms
        self.batch_norm1 = nn.BatchNorm2d(num_features=32)
        self.batch_norm2 = nn.BatchNorm2d(num_features=64)
        self.batch_norm3 = nn.BatchNorm2d(num_features=128)
        self.batch_norm4 = nn.BatchNorm2d(num_features=256)
        self.batch_norm5 = nn.BatchNorm2d(num_features=512)

        # pooling layer(s)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # fully connected layer(s)
        self.fc1 = nn.Linear(in_features=2048, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=102)

        # dropout regularisation
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = self.pool(x)

        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.pool(x)

        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = self.pool(x)

        x = F.relu(self.batch_norm5(self.conv5(x)))
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x
