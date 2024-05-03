import csv
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 64

transform = transforms.Compose(
    [transforms.ToTensor(),         # Transform the data into a tensor.
     #transforms.Resize((64, 64)), # Resize every image to be 500x500.
     transforms.RandomResizedCrop(size=(128, 128), antialias=True),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     ])

train_data = Flowers102(
    root="data",
    split="train",
    download=True,
    transform=transform
    )

val_data = Flowers102(
    root="data",
    split="val",
    download=True,
    transform=transform
    )

test_data = Flowers102(
    root="data",
    split="test",
    download=True,
    transform=transform
    )


train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# Names for each label from (line 25) since as far as I can tell pytorch does not seem to have this data included.
# https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/oxford_flowers102/oxford_flowers102_dataset_builder.py
with open("data/flowers-names.csv", "r") as f:
    reader = csv.reader(f)
    names = list(reader)

names = np.array(names).squeeze()

