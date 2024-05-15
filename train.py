import torch
import torch.nn as nn
import torch.optim as optim
from imlo_coursework.cnn import CNN
from imlo_coursework.load_data import device, train_dataloader, val_dataloader, cutmix_or_mixup

network = CNN()
# Move the network to the device to allow for CUDA support for much faster running times.
network = network.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
epochs = 750


def train_one_epoch():
    """
    Function which trains the network for one epoch. Expects to be ran inside a loop of multiple epochs
    to get a good result.
    """
    network.train()

    running_loss = 0
    running_accuracy = 0
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Store the pre cutmix_or_mixup values to calculate the training accuracy with.
        original_inputs = inputs
        original_labels = labels
        
        # Train the model on cutmix_or_mixup values.
        inputs, labels = cutmix_or_mixup(inputs, labels)
        
        # zero the gradients.
        optimizer.zero_grad()
        # predict outputs based on the inputs.
        outputs = network(inputs)
        # calculate the loss.
        loss = loss_fn(outputs, labels)
        loss.backward()
        # step the optimizer.
        optimizer.step()
        # training statistic, record the loss
        running_loss += loss.item()
        
        # calculating accuracy with cutmix_or_mixup is complicated, so just calculate the accuracy
        # the same way we do in validate_one_epoch, without training based on this data 
        with torch.no_grad():
            original_outputs = network(original_inputs)
            running_accuracy += torch.mean((torch.argmax(original_outputs, 1) == original_labels).float())
        
    total = len(train_dataloader)
    loss = running_loss / total
    accuracy = 100 * (running_accuracy / total)

    print(f"Training   - Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")


def validate_one_epoch():
    """
    Function which calculates the validation accuracy for one epoch. Excepts to be ran in the training loop to 
    validate the model on unseen data.
    """
    network.eval()

    running_loss = 0
    running_accuracy = 0
    for inputs, labels in val_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # We do not need gradients if we are not training.
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
torch.save(network.state_dict(), 'model.pt')
