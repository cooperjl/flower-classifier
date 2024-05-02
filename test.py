import torch
import torch.nn as nn
from imlo_coursework.cnn import CNN
from imlo_coursework.load_data import test_dataloader, device

model = CNN(classes=102)
model.load_state_dict(torch.load("model.pth"))

model = model.to(device)
model.eval()

loss_fn = nn.CrossEntropyLoss()

correct = 0
total = 0
for (inputs, labels) in test_dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()


accuracy = 100 * (correct / total)
print(f"Accuracy of model: {accuracy:.2f}%")

