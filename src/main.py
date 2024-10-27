#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import numpy as np
import os

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define optimizers
optimizers = {
    "Adam": optim.Adam,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop,
    "AdamW": optim.AdamW,
    "AdaGrad": optim.Adagrad,
}

# Define the neural network
class SimpleANN(nn.Module):
    def __init__(self, output_dim=10):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Convert labels to one-hot encoding for BinaryCrossEntropyLoss
def to_one_hot(labels, num_classes=10):
    return F.one_hot(labels, num_classes).float()

# Training and validation process
def train_and_validate(model, optimizer, criterion, train_loader, val_loader, epochs, lr_scheduler, patience, folder_name):
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    learning_rates = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_accuracy)

        # Validation phase
        val_loss, val_accuracy = test(model, criterion, val_loader, False)
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

        # Save learning rate and update scheduler
        learning_rates.append(optimizer.param_groups[0]['lr'])
        lr_scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(folder_name, 'best_model.pth'))
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    # Save plots for losses and accuracy
    plot_metrics(train_losses, val_losses, 'Loss', folder_name)
    plot_metrics(train_accs, val_accs, 'Accuracy', folder_name)
    plot_learning_rate(learning_rates, folder_name)
    plot_confusion_matrix(model, val_loader, folder_name)

def test(model, criterion, data_loader, is_test):
    """Validation or testing function."""
    model.eval()
    loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = loss / len(data_loader)
    accuracy = 100 * correct / total

    if is_test:
        print(f"Test Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def plot_metrics(train_values, val_values, metric_name, folder_name):
    """Plot and save training and validation metrics (loss or accuracy)."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_values, label='Training')
    plt.plot(val_values, label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(f'Training and Validation {metric_name}')
    plt.savefig(os.path.join(folder_name, f"{metric_name}.png"))
    plt.close()

def plot_learning_rate(learning_rates, folder_name):
    """Plot and save learning rate across epochs."""
    plt.figure(figsize=(8, 6))
    plt.plot(learning_rates, label='Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.savefig(os.path.join(folder_name, "Learning_Rate.png"))
    plt.close()

def plot_confusion_matrix(model, data_loader, folder_name):
    """Plot and save confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(folder_name, "Confusion_Matrix.png"))
    plt.close()

def data_preprocessing(batch_size):
    """Prepare data loaders."""
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    train_size = 50000
    val_size = 10000
    train_dataset, val_dataset = random_split(mnist_dataset, [train_size, val_size])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def main():
    batch_size = 64
    epochs = 20
    patience = 5
    initial_learning_rate = 0.001

    train_loader, val_loader, test_loader = data_preprocessing(batch_size)

    for opt_name, opt_func in optimizers.items():
        folder_name = f"{opt_name}_CrossEntropy"
        os.makedirs(folder_name, exist_ok=True)

        model = SimpleANN().to(device)
        optimizer = opt_func(model.parameters(), lr=initial_learning_rate)
        criterion = nn.CrossEntropyLoss()
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        print(f"\nStarting training with Optimizer: {opt_name}")
        train_and_validate(model, optimizer, criterion, train_loader, val_loader, epochs, lr_scheduler, patience, folder_name)

        print(f"\nTesting with Optimizer: {opt_name}")
        model.load_state_dict(torch.load(os.path.join(folder_name, 'best_model.pth')))
        test(model, criterion, test_loader, is_test=True)

if __name__ == "__main__":
    main()

