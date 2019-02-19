import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, size=(28,28)):
        super(ConvNet, self).__init__()
        self.name = self.__class__.__name__
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 10, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
        )
        linear_in = 20*((torch.tensor(size)//2)//2).prod()
        self.classifier = nn.Sequential(
            nn.Linear(linear_in,50),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(50,num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
