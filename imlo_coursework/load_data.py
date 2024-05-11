import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 16

crop_size = (128, 128)

# values outputted from normal_values.py
mean =[0.4702, 0.3985, 0.3176]
std = [0.2599, 0.2085, 0.2217]

# transforms.Resize(x) resizes the shortest side of the image to x pixels if x is an integer instead of a tuple.
# this allows us to maintain the aspect ratio of the image without zooming in in transform_val_test.
transform_train = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize(size=160, antialias=True),
    transforms.RandomCrop(size=crop_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=mean, std=std),
])

transform_val_test = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize(size=128, antialias=True),
    transforms.CenterCrop(size=crop_size),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=mean, std=std),
])

train_data = Flowers102(
    root="data",
    split="train",
    download=True,
    transform=transform_train
)

val_data = Flowers102(
    root="data",
    split="val",
    download=True,
    transform=transform_val_test
)

test_data = Flowers102(
    root="data",
    split="test",
    download=True,
    transform=transform_val_test
)


train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

