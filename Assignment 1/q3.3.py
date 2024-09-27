import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

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
### Data Loading
##########################

transform = transforms.ToTensor()

# Load FashionMNIST dataset
train_dataset = FashionMNIST(root='data', train=True, transform=transform, download=True)
test_dataset = FashionMNIST(root='data', train=False, transform=transform)

##########################
### Model and Hyperparameters
##########################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = InceptionNet(num_classes=10).to(device)

# Cross entropy loss function
cost_fn = torch.nn.CrossEntropyLoss()

# Fixed learning rate (use lr_max from the previous part)
lr_max = 10e-2

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=lr_max)

# Batch sizes (starting from 32 and going up to 8192 in powers of 2)
batch_sizes = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

# Hyperparameters
num_epochs = 1  # For simplicity, we'll run a short experiment with 1 epoch per batch size increment

# Collect stats for plotting
train_losses = []

##########################
### Helper Function
##########################

def compute_training_loss(model, data_loader):
    running_loss = 0.0
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.to(device), targets.to(device)
            logits = model(features)
            loss = cost_fn(logits, targets)
            running_loss += loss.item()
    model.train()  # Switch back to train mode
    return running_loss / len(data_loader)

##########################
### Training Loop
##########################

for batch_size in batch_sizes:
    print(f"Training with batch size: {batch_size}")
    
    # Create DataLoader with the new batch size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            features, targets = features.to(device), targets.to(device)
            
            # Forward pass
            logits = model(features)
            loss = cost_fn(logits, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Compute average training loss for this epoch
        avg_train_loss = running_loss / len(train_loader)
    
    # After 15 epochs for the current batch size, store the loss
    train_losses.append(avg_train_loss)
    
    # Output the training loss for this batch size
    print(f"Batch size {batch_size}, Training Loss: {avg_train_loss:.4f}")

##########################
### Plotting the Results
##########################

# Plot Training Loss for Different Batch Sizes
plt.figure(figsize=(10, 5))
plt.plot(batch_sizes, train_losses, marker='o', label='Training Loss')
plt.xscale('log')  # Use a log scale for the batch size axis
plt.xlabel('Batch Size (log scale)')
plt.ylabel('Training Loss')
plt.title('Training Loss vs Batch Size')
plt.grid(True)
plt.legend()
plt.show()