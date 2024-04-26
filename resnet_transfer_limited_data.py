import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset

# Define transformations for the MNIST dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet input size
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset with only the first two classes
mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# Filter the dataset to use only the first 50 data samples in each class
mnist_train_filtered = []
mnist_test_filtered = []

for i in range(2):  # First two classes
    indices_train = (mnist_train.targets == i).nonzero().squeeze(1)[:20]  # Get indices of the first 20 data samples
    indices_test = (mnist_test.targets == i).nonzero().squeeze(1)[:500]
    mnist_train_filtered.extend(Subset(mnist_train, indices_train))
    mnist_test_filtered.extend(Subset(mnist_test, indices_test))

class0 = []
indice0 = (mnist_train.targets == 0).nonzero().squeeze(1)[:20]
class0.extend(Subset(mnist_train, indice0))

class1 = []
indice1 = (mnist_train.targets == 1).nonzero().squeeze(1)[:20]
class1.extend(Subset(mnist_train, indice1))

# Define data loaders
train_loader = DataLoader(dataset=mnist_train_filtered, batch_size=5, shuffle=True)
test_loader = DataLoader(dataset=mnist_test_filtered, batch_size=64, shuffle=False)

# Load pre-trained ResNet model
resnet = torchvision.models.resnet18(pretrained=True)

# Modify the first convolutional layer to accept input with 1 channel
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Freeze all layers except the last two
for param in resnet.parameters():
    param.requires_grad = False

# Modify the last layer for binary classification
num_ftrs = resnet.fc.in_features

resnet.fc = nn.Sequential(
    nn.Linear(num_ftrs, 10),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(10, 1),  # Output 1 unit for binary classification
    nn.Sigmoid()  # Sigmoid activation for binary classification
)

# Get the second-to-last layer (before the final fully connected layer)
second_to_last_layer = nn.Sequential(*list(resnet.children())[:-1])

# Move model to device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)

# Define loss function and optimizer for binary classification
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)

# Training loop
num_epochs = 5
total_step = 8
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)  # Convert labels to float and unsqueeze to match output shape

        # Forward pass
        outputs = resnet(images)
        loss = criterion(outputs, labels)

        # second_to_last_output = second_to_last_layer(images)
        # # Reshape second_to_last_output before passing it to the first linear layer
        # second_to_last_output = second_to_last_output.view(second_to_last_output.size(0), -1)
        # # Pass the reshaped second_to_last_output to the first linear layer
        # second_to_last_output = resnet.fc[0](second_to_last_output)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 1 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

class0_loader = DataLoader(dataset=class0, batch_size=20, shuffle=False)
class1_loader = DataLoader(dataset=class1, batch_size=20, shuffle=False)

for image, label in class0_loader:
    images0 = image

for image, label in class1_loader:
    images1 = image

feature_0 = second_to_last_layer(images0)
feature_0 = feature_0.view(feature_0.size(0), -1)
feature_0 = resnet.fc[0](feature_0)
feature_1 = second_to_last_layer(images1)
feature_1 = feature_1.view(feature_1.size(0), -1)
feature_1 = resnet.fc[0](feature_1)

print(feature_0)
print(feature_1)

# Test the model
resnet.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        outputs = resnet(images)
        predicted = (outputs > 0.5).float()  # Convert probabilities to binary predictions
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {:.2f} %'.format(100 * correct / total))
