import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

batch_size = 16

size = 160
#size = (160, 160)
crop_size = (128, 128)

transform_train = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize(size=size, antialias=True),
    transforms.RandomCrop(size=crop_size),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandAugment(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val_test = transforms.Compose([
    transforms.PILToTensor(),
    transforms.Resize(size=size, antialias=True),
    transforms.CenterCrop(size=crop_size),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

