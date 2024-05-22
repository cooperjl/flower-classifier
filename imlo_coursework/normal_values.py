import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
from imlo_coursework.load_data import batch_size, crop_size

def calculate_values_for_normalise():
    # Using a custom adaptation of the transform_val_test transform, since we do not want to normalise here.
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize(size=128, antialias=True),
        transforms.CenterCrop(size=crop_size),
        transforms.ToDtype(torch.float32, scale=True),
    ])
    # Get the train data and apply the custom transform.
    train_data = Flowers102(
        root="data",
        split="train",
        download=True,
        transform=transform
    )

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    running_mean = 0
    running_std = 0

    for inputs, _ in train_dataloader:
        # reshape the tensor from (batch_size, 3, 128, 128) to (batch_size, 3, 128*128)
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        # calculate the mean and std over the 128*128 dimension which includes all the values of the images
        # sum those values and add them to running value
        running_mean += inputs.mean(dim=2).sum(dim=0)
        running_std += inputs.std(dim=2).sum(dim=0)

    # Print the mean and std values for manual copying into load_data.py. This is to avoid having to rerun the 
    # whole normalisation function every time the transform is called as this would be needless overhead.
    # divide by 1020 since that is the total number of images in the training set
    print(f"mean: {running_mean/1020}")
    print(f"std: {running_std/1020}")

