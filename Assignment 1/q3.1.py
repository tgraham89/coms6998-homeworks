# Reference used:
# https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/tricks/cyclical-learning-rate.ipynb

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST



if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True

##########################
### MODEL
##########################


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
    

##########################
### SETTINGS
##########################

batch_size = 64  # As specified
transform = transforms.ToTensor()

# Load FashionMNIST dataset
train_dataset = FashionMNIST(root='data', train=True, transform=transform, download=True)
test_dataset = FashionMNIST(root='data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

##########################
### Model and Hyperparameters
##########################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = InceptionNet(num_classes=10).to(device)

# Cross entropy loss function
cost_fn = torch.nn.CrossEntropyLoss()

# Candidate learning rates as requested
learning_rates = [10e-9, 10e-8, 10e-7, 10e-6, 10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e1]

# Hyperparameters
num_epochs = 5
num_chunks = 10  # We divide 5 epochs into 10 chunks (half an epoch per learning rate)
iter_per_chunk = len(train_loader) // num_chunks

# Store accuracies for plotting
accuracies = []

##########################
### Helper Function
##########################

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

##########################
### Training Loop
##########################

for lr in learning_rates:
    # Define optimizer with the current learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    
    print(f'Training with learning rate: {lr}')
    
    for chunk in range(iter_per_chunk):
        print(f'Chunk {chunk + 1} of {iter_per_chunk}')
        for batch_idx, (features, targets) in enumerate(train_loader):
            if batch_idx >= iter_per_chunk:  # Train for half an epoch
                break
            
            features, targets = features.to(device), targets.to(device)
            
            # Forward pass
            logits = model(features)
            loss = cost_fn(logits, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Compute validation accuracy after this learning rate chunk
    val_acc = compute_accuracy(model, val_loader)
    accuracies.append(val_acc.item())  # Store accuracy for this learning rate
    
    print(f'Validation accuracy for lr={lr}: {val_acc.item():.2f}%')

##########################
### Plotting the Results
##########################

plt.plot(learning_rates, accuracies, marker='o')
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Validation Accuracy (%)')
plt.title('Learning Rate vs Validation Accuracy')
plt.grid(True)
plt.show()