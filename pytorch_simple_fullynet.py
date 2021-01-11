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


# Create Fully Connected Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes): # 28 x 28 = 784
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])



# model = NN(784, 10)
# x = torch.rand(64, 784) # 64 is the mini-batch size
# print(model(x).shape)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10
load_model = True

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # shuffle batches
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True) # shuffle batches

# Initialize network
model = NN(input_size=input_size, num_classes=num_classes).to(device) # cuda or cpu

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'))


# Train Network
for epoch in range(num_epochs):
    losses = []

    if epoch %3 == 0:
        print('yes')
        checkpoint = {'state_dict' : model.state_dict(), 'optimizer' : optimizer.state_dict()}
        save_checkpoint(checkpoint)
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # print(data.shape)
        #  get to correct shape
        data = data.reshape(data.shape[0], -1)
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
    num_correct = 0
    num_samples = 0
    model.eval()

    if loader.dataset.train:
        print('Checking accuracy on training data')
    else:
        print('Checking accuracy on test data')

    with torch.no_grad(): # we dont have to calculate gradient when we check accuracy
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)
            # print(x.shape)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(float(num_correct)/float(num_samples))
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)