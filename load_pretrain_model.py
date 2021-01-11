# We apply checkpoint in this simple NN, by doing this, we can continue our work based on the previously saved pth.tar file
#  which can be confirm by the fact that when we run this twice, at the beginning of the second time we have the last loss value in the first time.
# This can be convenient when we what a small loss.

# Imports
import torch
import torch.nn as nn
import torch.optim as optim #gradient descent
import torch.nn.functional as F # relu tanh
from torch.utils.data import DataLoader #create minibatch
import torchvision
import torchvision.datasets as datasets
# we use MNIST dataset
import torchvision.transforms as transforms

# NOT USE IN THIS SECTION
# class CNN(nn.Module):
#     def __init__(self, in_channels, num_classes):
#         super(CNN, self).__init__()
#         self.conv1 = nn. Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
#         self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3 ,3), stride=(1, 1), padding=(1, 1))
#         self.fc1 = nn.Linear(16*7*7, num_classes)
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc1(x)
#         return x
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 3
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 5


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


# Load pretrain model and then modify it
model = torchvision.models.vgg16(pretrained=True)
print(model)


# Transfer learning
# we want only use backprop in some of the layers, dont change weight of the layers, would not change anything
# By applying this, we will only trained the last few layers written below (model.avgpool and model.classifier)
for param in model.parameters():
    param.requires_grad = False


#  we want to remove the avgpool and also change the classifier, by creating this class, we can do it.
model.avgpool = Identity() #meaning that we jump the average pooling step
# model.classifier = nn.Linear(512, 10) # we change the classifier
# model.classifier[0] = nn.Linear(512, 10) # if I wrote like this I can change specific layer in classifier, check it out by replacing the above line

#  Or we can write as this we have two linear layer to have the classifier
model.classifier = nn.Sequential(nn.Linear(512, 100),
                                 nn.ReLU(),
                                 nn.Linear(100, 10))
#  Or we can write as if we want to keep the rest of the layers
# for i in range(1,7):
#     model.classifier[i] = Identity()

print(model)
model.to(device)

# Load Data
train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # shuffle batches
# test_dataset = datasets.CIFAR10(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True) # shuffle batches


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # print(data.shape)
        #  get to correct shape
        # data = data.reshape(data.shape[0], -1)
        # print(data.shape)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)
        losses.append(loss.item())
        # Backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
    mean_loss = sum(losses)/len(losses)
    print(f'Loss at epoch {epoch} was {mean_loss:.5f}')

# Check the accuracy on training $ test to see how good our model

def check_accuracy(loader, model):

    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on test data')
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad(): # we dont have to calculate gradient when we check accuracy
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            # x = x.reshape(x.shape[0], -1)
            print(x.shape)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(float(num_correct)/float(num_samples))
    model.train()

check_accuracy(train_loader, model)
# check_accuracy(test_loader, model)