import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional layers with a kernel (filter) size of 3x3.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)

        # Batch normalisation layers.
        self.batch_norm1 = nn.BatchNorm2d(num_features=32)
        self.batch_norm2 = nn.BatchNorm2d(num_features=64)
        self.batch_norm3 = nn.BatchNorm2d(num_features=128)
        self.batch_norm4 = nn.BatchNorm2d(num_features=256)
        self.batch_norm5 = nn.BatchNorm2d(num_features=512)

        # Pooling layer is reusable for every pool layer in the model, since it uses the same parameters.
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers are used to combine features into the 102 classes for Flowers102.
        self.fc1 = nn.Linear(in_features=2048, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=2048)
        self.fc3 = nn.Linear(in_features=2048, out_features=102)

        # Dropout regularisation randomly zeroes some features to decrease overfitting and increase robustness.
        self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        # Go through layers of the model. This implements what is shown in the diagram in the report.

        # Conv, followed by BatchNorm, followed by ReLU.
        x = F.relu(self.batch_norm1(self.conv1(x)))
        # MaxPool
        x = self.pool(x)

        x = F.relu(self.batch_norm2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = self.pool(x)

        x = F.relu(self.batch_norm4(self.conv4(x)))
        x = self.pool(x)

        x = F.relu(self.batch_norm5(self.conv5(x)))
        x = self.pool(x)
        
        # Flatten for input into fully connected layers.
        x = torch.flatten(x, 1)
        # Fully connected, followed by ReLU, followed by dropout.
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        # output features should not have an activation function and especially not dropout.
        x = self.fc3(x)

        return x
