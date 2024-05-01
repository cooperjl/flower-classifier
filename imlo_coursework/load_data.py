import csv
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102

# Following https://pytorch.org/tutorials/beginner/basics/data_tutorial.html 
# to understand how to load the dataset.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((300, 300)),
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


train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
val_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)


# Names for each label from (line 25) since as far as I can tell pytorch does not seem to have this data included.
# https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/oxford_flowers102/oxford_flowers102_dataset_builder.py
with open("data/flowers-names.csv", "r") as f:
    reader = csv.reader(f)
    names = list(reader)

names = np.array(names).squeeze()

# pandas way, but unsure whether allowed pandas
# names = pd.read_csv("data/flowers-names.csv", header=None).squeeze()

