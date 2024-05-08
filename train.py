import torch
import torch.nn as nn
import torch.optim as optim
from imlo_coursework.cnn import CNN
from imlo_coursework.load_data import device, train_dataloader, val_dataloader

network = CNN()
network = network.to(device)

loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(network.parameters(), lr=0.001, weight_decay=0.001)
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
epochs = 300 


def train_one_epoch():
    network.train(True)

    running_loss = 0
    running_accuracy = 0
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = network(inputs)
        loss = loss_fn(outputs, labels)

        running_loss += loss.item()
        running_accuracy += torch.mean((torch.argmax(outputs, 1) == labels).float())

        loss.backward()
        optimizer.step()

    total = len(train_dataloader)
    loss = running_loss / total
    accuracy = 100 * (running_accuracy / total)

    print(f"Training   - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")


def validate_one_epoch():
    network.train(False)

    running_loss = 0
    running_accuracy = 0
    for inputs, labels in val_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = network(inputs)
            loss = loss_fn(outputs, labels)
            
            running_loss += loss.item()
            running_accuracy += torch.mean((torch.argmax(outputs, 1) == labels).float())

    total = len(val_dataloader)
    loss = running_loss / total
    accuracy = 100 * (running_accuracy / total)

    print(f"Validation - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")


for epoch in range(epochs):
    print(f"Epoch {epoch+1}:")

    train_one_epoch()
    validate_one_epoch()


print("Finished training")
torch.save(network.state_dict(), 'model.pth')
