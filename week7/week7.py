import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# -------------------------------
# 1. Download dataset with kagglehub
# -------------------------------
path = kagglehub.dataset_download("arnaudeq/cats-vs-dogs-1000")
print("Path to dataset files:", path)

# The dataset usually has a structure like:
# path/train/cats, path/train/dogs, path/test/cats, path/test/dogs

# -------------------------------
# Inspect the downloaded directory structure (for debugging/understanding)
# -------------------------------
print("\nDirectory contents:")
for root, dirs, files in os.walk(path):
    print(root)
    for d in dirs:
        print(os.path.join(root, d))
    for f in files:
        print(os.path.join(root, f))


# -------------------------------
# 2. Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 3. Data transforms
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # resize to fit ResNet input
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],   # ImageNet mean
                         [0.229, 0.224, 0.225])   # ImageNet std
])

# Update the path to the training directories based on inspection
# Based on the directory listing, the actual path is nested within another folder.
train_data_path = os.path.join(path, "dogs_cats_sample_1000", "dogs_cats_sample_1000", "train") # Corrected path based on inspection

print(f"\nAttempting to load training data from: {train_data_path}")
# Check if the training directory exists and contains class subdirectories
if not os.path.exists(train_data_path):
    print(f"Error: Training data path not found at {train_data_path}")
elif not any(os.path.isdir(os.path.join(train_data_path, d)) for d in os.listdir(train_data_path)):
     print(f"Error: Training data path {train_data_path} does not contain class subdirectories.")
else:
    print("Training data path looks correct.")


# Load training dataset
train_dataset = datasets.ImageFolder(root=train_data_path, transform=transform)

# Create data loader for training
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("Classes:", train_dataset.classes)  # should print ['cats', 'dogs']

# Note: The 'test' directory in this dataset appears to be empty, causing FileNotFoundError with ImageFolder.
# We will proceed with training using only the training data for now.
# If you have a separate test set or would like to use a portion of the training data for validation,
# we can modify the code to include evaluation later.

# -------------------------------
# 4. Load pretrained ResNet18
# -------------------------------
model = models.resnet18(pretrained=True)

# Freeze feature extractor parameters
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer for 2-class classification (cats vs dogs)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

# -------------------------------
# 5. Loss function and optimizer
# -------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001) # Only optimize the parameters of the new final layer

# -------------------------------
# 6. Training loop
# -------------------------------
epochs = 5
print("\nStarting training...")
for epoch in range(epochs):
    model.train() # Set model to training mode
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

print("\nTraining finished.")

# -------------------------------
# 7. Evaluation - Removed due to empty test directory in the dataset
# If you have a test set, you can add evaluation code here.
# For example, load a test dataset and use model.eval() and torch.no_grad()
# to calculate accuracy on the test set.
# -------------------------------
# print("\nStarting evaluation...")
# model.eval() # Set model to evaluation mode
# correct = 0
# total = 0
# with torch.no_grad(): # Disable gradient calculation for evaluation
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.to(device), labels.to(device)
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f"Test Accuracy: {100 * correct / total:.2f}%")
