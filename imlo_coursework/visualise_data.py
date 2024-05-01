import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision.transforms import ToTensor
from torchvision.datasets import Flowers102
from imlo_coursework.load_data import train_data, names
"""
Function to plot a 3x3 grid of images from the dataset, labelled with their correct classifications.
"""
def grid_of_flowers():
    # Plotting a random selection of images using matplotlib
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(train_data), size=(1,)).item()
        img, label = train_data[int(sample_idx)]
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        plt.title(names[int(label)])

        # pytorch uses CxHxW, whereas matplotlib uses HxWxC, so we need to fix that.
        plt.imshow(img.permute(1,2,0))
        print(img)

    plt.show()

def grid_of_flowers2(img, pred, label):
    # Plotting a random selection of images using matplotlib
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        # sample_idx = torch.randint(len(input), size=(1,)).item()
        #img, label = inputs[int(sample_idx)]
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        plt.title(f"actual: {names[int(label)]}\n pred: {names[int(pred)]}")

        # pytorch uses CxHxW, whereas matplotlib uses HxWxC, so we need to fix that.
        plt.imshow(img.permute(1,2,0))

    # plt.show()

if __name__ == "__main__":
    grid_of_flowers()
