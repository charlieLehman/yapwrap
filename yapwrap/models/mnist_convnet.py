import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

__all__ = ['MNIST_ConvNet', 'FASHION_MNIST_ConvNet']

class MNIST_ConvNet(nn.Module):
    def __init__(self, in_channels=1):
        super(MNIST_ConvNet, self).__init__()
        self.name = self.__class__.__name__
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.drop1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv1_bn(x)
        x = F.relu(F.max_pool2d(self.drop1(self.conv2(x)), 2))
        x = self.conv2_bn(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

## Same as MNIST_ConvNet for now
class FASHION_MNIST_ConvNet(nn.Module):
    def __init__(self, in_channels=1):
        super(FASHION_MNIST_ConvNet, self).__init__()
        self.name = self.__class__.__name__
        self.conv1 = nn.Conv2d(in_channels, 10, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.drop1 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = self.conv1_bn(x)
        x = F.relu(F.max_pool2d(self.drop1(self.conv2(x)), 2))
        x = self.conv2_bn(x)
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
