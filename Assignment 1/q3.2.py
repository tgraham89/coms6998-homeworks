# Reference used:
# https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/tricks/cyclical-learning-rate.ipynb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# Model
class InceptionModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, out_channels3, out_channels4):
        super(InceptionModule, self).__init__()
        
        # 1x1 convolution
        self.branch1 = torch.nn.Conv2d(in_channels, out_channels1, kernel_size=1)
        
        # 1x1 convolution followed by 3x3 convolution
        self.branch2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels2, kernel_size=1),
            torch.nn.Conv2d(out_channels2, out_channels2, kernel_size=3, padding=1)
        )
        
        # 1x1 convolution followed by 5x5 convolution
        self.branch3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels3, kernel_size=1),
            torch.nn.Conv2d(out_channels3, out_channels3, kernel_size=5, padding=2)
        )
        
        # 3x3 max pooling followed by 1x1 convolution
        self.branch4 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels, out_channels4, kernel_size=1)
        )

    def forward(self, x):
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)
        branch3_out = self.branch3(x)
        branch4_out = self.branch4(x)
        
        outputs = [branch1_out, branch2_out, branch3_out, branch4_out]
        return torch.cat(outputs, 1)


class InceptionNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(InceptionNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception1 = InceptionModule(64, 32, 32, 32, 32)
        self.inception2 = InceptionModule(128, 64, 64, 64, 64)
        
        self.pool2 = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.inception1(x)
        x = self.inception2(x)
        
        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# Settings
batch_size = 64  # As specified
transform = transforms.ToTensor()

# Load FashionMNIST dataset
train_dataset = FashionMNIST(root='data', train=True, transform=transform, download=True)
test_dataset = FashionMNIST(root='data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = InceptionNet(num_classes=10).to(device)

# Cross entropy loss function
cost_fn = torch.nn.CrossEntropyLoss()

# Learning rate parameters
lr_min = 10e-4
lr_max = 10e-2

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr_min)

# Cyclical learning rate scheduler (exponential mode)
scheduler = torch.optim.lr_scheduler.CyclicLR(
    optimizer, 
    base_lr=lr_min, 
    max_lr=lr_max, 
    step_size_up=5 * len(train_loader) // 2,  # Steps for half an epoch
    mode='exp_range', 
    gamma=0.99994,  # Decay factor for exponential decay
    cycle_momentum=False  # Disable momentum cycling
)

# Hyperparameters
num_epochs = 150

# Collect stats for plotting
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Helper function to compute accuracy
def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            probas = F.softmax(logits, dim=1)
            _, predicted_labels = torch.max(probas, 1)
            num_examples += targets.size(0)
            correct_pred += (predicted_labels == targets).sum()
    model.train()  # Switch back to train mode
    return correct_pred.float() / num_examples * 100

# Training Loop
for epoch in range(num_epochs):
    model.train()  # Ensure model is in training mode
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for batch_idx, (features, targets) in enumerate(train_loader):
        features, targets = features.to(device), targets.to(device)
        
        # Forward pass
        logits = model(features)
        loss = cost_fn(logits, targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate

        # Collect statistics
        running_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()
        
    # Calculate average loss and accuracy for this epoch
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # Evaluate on validation set
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    model.eval()
    with torch.no_grad():
        for features, targets in val_loader:
            features, targets = features.to(device), targets.to(device)
            logits = model(features)
            loss = cost_fn(logits, targets)
            val_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total_val += targets.size(0)
            correct_val += (predicted == targets).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100 * correct_val / total_val
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')

# Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot Training and Validation Accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs+1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
