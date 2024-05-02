import torch
import torch.nn as nn
import torch.optim as optim
from imlo_coursework.cnn import CNN
from imlo_coursework.load_data import train_dataloader, device

network = CNN(classes=102)
network = network.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=0.001)
epochs = 25

for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(train_dataloader, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = network(inputs)
        loss = loss_fn(outputs, labels)

        if i % 100 == 0:
            print(f"epoch: {epoch}, loss: {loss}")

        loss.backward()
        optimizer.step()


print("Finished training")
torch.save(network.state_dict(), 'model.pth')
