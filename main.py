import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# GPU check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 64
initial_learning_rate = 0.001
num_epochs = 20
patience = 5  # Early stopping patience

# Data preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split dataset into training, validation, and test sets
train_size = 50000
val_size = 10000

train_dataset, val_dataset = random_split(mnist_dataset, [train_size, val_size])
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the neural network model
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
        return torch.sigmoid(self.fc3(x)) if criterion_type == "BinaryCrossEntropy" else self.fc3(x)

# Convert labels to one-hot encoding for BinaryCrossEntropyLoss
def to_one_hot(labels, num_classes=10):
    return F.one_hot(labels, num_classes).float()

# Trainer class
class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, combo_name):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.combo_name = combo_name
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.learning_rates = []

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in self.train_loader:
            images, labels = images.to(device), labels.to(device)

            if criterion_type == "BinaryCrossEntropy":
                labels = to_one_hot(labels).to(device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if criterion_type == "BinaryCrossEntropy":
                correct += (predicted == labels.argmax(dim=1)).sum().item()
            else:
                correct += (predicted == labels).sum().item()

        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        return running_loss / len(self.train_loader), correct / total

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(device), labels.to(device)
                
                if criterion_type == "BinaryCrossEntropy":
                    labels = to_one_hot(labels).to(device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                if criterion_type == "BinaryCrossEntropy":
                    correct += (predicted == labels.argmax(dim=1)).sum().item()
                else:
                    correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy() if criterion_type != "BinaryCrossEntropy" else labels.argmax(dim=1).cpu().numpy())

        return val_loss / len(self.val_loader), correct / total, all_preds, all_labels

    def train_and_validate(self):
        train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
        for epoch in range(num_epochs):
            train_loss, train_accuracy = self.train_one_epoch()
            val_loss, val_accuracy, val_preds, val_labels = self.validate()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            self.scheduler.step(val_loss)

            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
            print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}\n")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), f'{self.combo_name}_best_model.pth')
            else:
                self.patience_counter += 1

            if self.patience_counter >= patience:
                print("Early stopping triggered.")
                break

            # Plot confusion matrix at the end of each epoch
            self.plot_confusion_matrix(val_labels, val_preds, epoch)

        self.save_plots(train_losses, val_losses, train_accuracies, val_accuracies)
        self.plot_learning_rate()
        return val_loss, val_accuracy

    def plot_confusion_matrix(self, labels, preds, epoch):
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title(f'Confusion Matrix for {self.combo_name} - Epoch {epoch + 1}')
        plt.savefig(f"{self.combo_name}_confusion_matrix_epoch_{epoch + 1}.png")
        plt.close()

    def save_plots(self, train_losses, val_losses, train_accuracies, val_accuracies):
        plt.figure()
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Loss for {self.combo_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{self.combo_name}_loss.png")
        plt.close()

        plt.figure()
        plt.plot(train_accuracies, label='Training Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.title(f'Accuracy for {self.combo_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f"{self.combo_name}_accuracy.png")
        plt.close()

    def plot_learning_rate(self):
        plt.figure()
        plt.plot(self.learning_rates, label='Learning Rate')
        plt.title(f'Learning Rate Schedule for {self.combo_name}')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.savefig(f"{self.combo_name}_learning_rate.png")
        plt.close()

# Main experiment loop
def main():
    best_accuracy = 0
    best_combo = None
    optimizers = {
    "Adam": optim.Adam,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop,
    "AdamW": optim.AdamW,
    "AdaGrad": optim.Adagrad,
}
    loss_functions = {
    "CrossEntropy": nn.CrossEntropyLoss(),
    "BinaryCrossEntropy": nn.BCELoss(),
}

    for opt_name, opt_func in optimizers.items():
        for loss_name, loss_func in loss_functions.items():
            global criterion_type
            criterion_type = loss_name

            if loss_name == "HingeLoss":
                print(f"Skipping incompatible combination: Optimizer: {opt_name}, Loss Function: HingeLoss")
                continue  # Skip Hinge loss

            model = SimpleANN().to(device)
            optimizer = opt_func(model.parameters(), lr=initial_learning_rate)
            criterion = loss_func if loss_name != "BinaryCrossEntropy" else nn.BCELoss()
            
            combo_name = f"{opt_name}_{loss_name}"
            print(f"Training with Optimizer: {opt_name} and Loss Function: {loss_name}")
            trainer = Trainer(model, optimizer, criterion, train_loader, val_loader, combo_name)
            val_loss, val_accuracy = trainer.train_and_validate()

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_combo = (opt_name, loss_name)

    print(f"Best combination: Optimizer: {best_combo[0]}, Loss Function: {best_combo[1]} with Validation Accuracy: {best_accuracy:.4f}")

if __name__ == "__main__":
    main()



