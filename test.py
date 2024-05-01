import torch
import torch.nn as nn
from imlo_coursework.cnn import CNN
from imlo_coursework.load_data import val_dataloader, test_dataloader, device
from imlo_coursework.visualise_data import grid_of_flowers2
import matplotlib.pyplot as plt

model = CNN(102)
model.load_state_dict(torch.load("model.pth"))


model = model.to(device)
model.eval()

loss_fn = nn.CrossEntropyLoss()

for data in test_dataloader:
    inputs, labels = data
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        print(loss)

        for i in range(len(inputs)):
            print(i) # TODO: make this work lol
            # grid_of_flowers2(inputs[i], outputs[i], labels[i])
            #print(outputs[i].item(), labels[i].item())


plt.show()
