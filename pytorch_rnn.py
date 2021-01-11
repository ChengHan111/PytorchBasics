# In this part we apply RNN, LSTM and GRU to MNIST network, RNN usually is not used for MNIST but it is just a simple example
# of how we can reach this

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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 2

# Create a RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # can also change to gru instead
        # self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        #  When apply to LSTM the following two status are required
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # out, _ = self.lstm(x, (h0,c0))


        # Forward Prop
        out, _ = self.rnn(x, h0)

        #  can also change to gru instead
        # out, _ = self.gru(x, h0)

        # when applying lstm
        # we remove the below line 'out = out.reshape(out.shape[0], -1)' and change the line 'out = self.fc(out)' into
        # out = self.fc(out[:,-1,:])

        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

# Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # shuffle batches
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True) # shuffle batches

# Initialize network
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device) # cuda or cpu

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device).squeeze(1)
        targets = targets.to(device=device)

        # Forward
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

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
            x = x.to(device=device).squeeze(1)
            # print(x.shape)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(float(num_correct)/float(num_samples))
    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)