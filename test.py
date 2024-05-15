import torch
import torch.nn as nn
from imlo_coursework.cnn import CNN
from imlo_coursework.load_data import test_dataloader, device

model = CNN()
model.load_state_dict(torch.load("model.pt"))

model = model.to(device)
model.eval()

loss_fn = nn.CrossEntropyLoss()

# Functionally the same as validation, just on the testing data instead.

running_loss = 0
running_accuracy = 0
for (inputs, labels) in test_dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        running_loss += loss.item()
        running_accuracy += torch.mean((torch.argmax(outputs, 1) == labels).float())

total = len(test_dataloader)
loss = running_loss / total
accuracy = 100 * (running_accuracy / total)

print(f"Testing - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

