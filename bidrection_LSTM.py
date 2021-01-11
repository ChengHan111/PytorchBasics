# BRNN的idea是将传统RNN的状态神经元拆分为两个部分，一个负责positive time direction(forward states)，另一个负责negative time direction(backward states)。
# Forward states的输出并不会连接到Backward states的输入。因此结构如下所示。如果没有backward layer的话就跟传统RNN相同了。
# https://zhuanlan.zhihu.com/p/34821214

# Imports
import torch
import torch.nn as nn
import torch.optim as optim  # gradient descent
import torch.nn.functional as F  # relu tanh
from torch.utils.data import DataLoader  # create minibatch
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


#  Create a bidrectional LSTM
class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # we time by 2 since one is going to go forward and one backward, we need to expand by 2 in this Tensor
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    # Load Data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  # shuffle batches
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)  # shuffle batches

# Initialize network
model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device)  # cuda or cpu

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

    with torch.no_grad():  # we dont have to calculate gradient when we check accuracy
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            # print(x.shape)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(float(num_correct) / float(num_samples))
    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)