import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import struct
import psutil

def read_idx(filename):
    """Function to read idx formatted files."""
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

def print_memory_usage():
    """Function to print current memory usage."""
    process = psutil.Process()
    mem = process.memory_info()
    print(f"Current Memory Usage: {mem.rss / (1024 * 1024):.2f} MB")  # Resident Set Size

# Load training images and labels
train_images = read_idx('Datasets\MNIST\train-images.idx3-ubyte')
train_labels = read_idx('Datasets\MNIST\train-labels.idx1-ubyte')

# Load test images and labels
test_images = read_idx('Datasets\MNIST\t10k-images.idx3-ubyte')
test_labels = read_idx('Datasets\MNIST\t10k-labels.idx1-ubyte')

# Normalize the images by dividing by 255 (grayscale)
train_images = torch.tensor(train_images / 255.0, dtype=torch.float32)
test_images = torch.tensor(test_images / 255.0, dtype=torch.float32)

# Convert labels to tensors
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Add a channel dimension to images (MNIST images are grayscale, so 1 channel)
train_images = train_images.unsqueeze(1)  # Shape: (N, 1, 28, 28)
test_images = test_images.unsqueeze(1)    # Shape: (N, 1, 28, 28)

# Create dataset and split into training and validation
full_train_dataset = data.TensorDataset(train_images, train_labels)
train_size = int(0.8 * len(full_train_dataset))  # 80% for training
val_size = len(full_train_dataset) - train_size  # 20% for validation
train_dataset, val_dataset = data.random_split(full_train_dataset, [train_size, val_size])

# Create test dataset
test_dataset = data.TensorDataset(test_images, test_labels)

# Create data loaders for batching
train_loader = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = data.DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = data.DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define the Convolutional Neural Network (CNN) model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Conv2D layers (as MNIST is 2D)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 1 input channel, 32 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Fully Connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 200)  # Flatten 64 channels of size 7x7 after pooling
        self.fc2 = nn.Linear(200, 10)  # Output layer with 10 classes (digits 0-9)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.5)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Convolutional layers with ReLU and pooling
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        
        # Flatten the output of the conv layers
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return torch.log_softmax(x, dim=1)

# Initialize the model, loss function, and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-7)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

# Training loop
epochs = 15
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    print(f"\nEpoch {epoch + 1}/{epochs}")
    
    
    for images, labels in train_loader:
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Step the learning rate scheduler
    scheduler.step()
    
    # Validation at the end of each epoch
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / len(train_loader):.4f}, '
          f'Val Loss: {val_loss / len(val_loader):.4f}, '
          f'Val Accuracy: {100 * correct / total:.2f}%')


# Final test evaluation
correct = 0
total = 0
with torch.no_grad():  # Disable gradient calculation
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Final Test Accuracy: {100 * correct / total:.2f}%')


