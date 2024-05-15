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

    mean = 0
    std = 0
    total = 0

    for inputs, _ in train_dataloader:
        batch_amount = inputs.size(0)
        inputs = inputs.view(batch_amount, inputs.size(1), -1)
        mean += inputs.mean(2).sum(0)
        std += inputs.std(2).sum(0)
        total += batch_amount

    # Print the mean and std values for manual copying into load_data.py. This is to avoid having to rerun the 
    # whole normalisation function every time the transform is called as this would be needless overhead.
    print(f"mean: {mean/total}")
    print(f"std: {std/total}")

