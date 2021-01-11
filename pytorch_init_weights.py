# This part tells how to initizalize weight in pytorch, pytorch has initial weights normally we do not have to worry about it but it
# we want to know how to change it, we can go through this code
# kaiming initialization ... blablabla

# Imports
import torch
import torch.nn as nn
import torch.optim as optim #gradient descent
import torch.nn.functional as F # relu tanh
from torch.utils.data import DataLoader #create minibatch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn. Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3 ,3), stride=(1, 1), padding=(1, 1))
        self.fc1 = nn.Linear(16*7*7, num_classes)

        self.initialize_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x

    def initialize_weights(self):
        for m in self.modules(): # all the above conv2d, maxpool2d blabla are saved in modules
            print(m) # we can get the entire network here.
            if isinstance(m, nn.Conv2d):
#                 we are now using kaiming
                nn.init.kaiming_uniform_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            #  These are some examples of how we initialize different layers, we do not have such layers below in our CNN, but it cam be used in other
            # different situation
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    model = CNN(in_channels=3, num_classes=10)