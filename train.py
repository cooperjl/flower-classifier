import torch.nn as nn
import torch.optim as optim
import torch
from imlo_coursework.cnn import CNN
from imlo_coursework.load_data import train_dataloader, device

network = CNN(classes=102)
# move the network to the device
network = network.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=0.001)
epochs = 25

losses = []
for epoch in range(epochs):
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # move the inputs and labels to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = network(inputs)
        loss = loss_fn(outputs, labels)

        if i % 100 == 0:
            print(f"epoch: {epoch}, loss: {loss}")


        loss.backward()
        optimizer.step()

print("finished training")
torch.save(network.state_dict(), 'model')
