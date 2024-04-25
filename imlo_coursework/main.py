import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToTensor
from torchvision.datasets import Flowers102

# Following https://pytorch.org/tutorials/beginner/basics/data_tutorial.html 
# to understand how to load the dataset.

train_data = Flowers102(
    root="data",
    split="train",
    download=True,
    transform=ToTensor()
)

test_data = Flowers102(
    root="data",
    split="test",
    download=True,
    transform=ToTensor()
)

# There is also a "var" split, which I'm unsure of what is useful for as of my current knowledge.

# Names for each label from (line 25) since as far as I can tell pytorch does not seem to have this data included.
# https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/datasets/oxford_flowers102/oxford_flowers102_dataset_builder.py
names = pd.read_csv("data/flowers-names.csv", header=None).squeeze()

# Plotting a random selection of images using matplotlib
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_data), size=(1,)).item()
    img, label = train_data[int(sample_idx)]
    figure.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.title(names[label])

    # pytorch uses CxHxW, whereas matplotlib uses HxWxC, so we need to fix that.
    plt.imshow(img.permute(1,2,0))

plt.show()
