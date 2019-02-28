import torchvision.transforms as tvtfs
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from yapwrap.dataloaders import *
import numpy as np
import torch
import os

class OOD_CIFAR10(CIFAR10):
    def __init__(self, root='./data', size=32, batch_sizes={'train':128,'test':100, 'ood':100}, transforms={'train':None, 'test':None, 'ood':None}, dataset_len=2500):
        super(OOD_CIFAR10, self).__init__(root, size, batch_sizes, transforms)
        self.ood_batch_size = batch_sizes['ood']
        if transforms['ood'] is None:
            self.ood_transform = self.test_transform
        else:
            self.ood_transform = transforms['ood']

        self.dataset_len = dataset_len

    def ood_iter(self):
        testset = dset.CIFAR10(root=self.root, train=False, download=True, transform=self.ood_transform)
        rs = torch.utils.data.RandomSampler(testset, True, self.dataset_len)
        ood_iter = DataLoader(testset, batch_size=self.ood_batch_size,
                               shuffle=False, num_workers=12, pin_memory=True, sampler=rs)
        ood_iter.metric_set = 'ood'
        ood_iter.name = self.name
        ood_iter.__len__ = self.dataset_len
        return ood_iter

    def __len__(self):
        return self.dataset_len

class OOD_CIFAR100(CIFAR100):
    def __init__(self, root='./data', size=32, batch_sizes={'train':128,'test':100, 'ood':100}, transforms={'train':None, 'test':None, 'ood':None}, dataset_len=2000):
        super(OOD_CIFAR100, self).__init__(root, size, batch_sizes, transforms)
        self.ood_batch_size = batch_sizes['ood']
        if transforms['ood'] is None:
            self.ood_transform = self.test_transform
        else:
            self.ood_transform = transforms['ood']

        self.dataset_len = dataset_len

    def ood_iter(self):
        testset = dset.CIFAR100(root=self.root, train=False, download=True, transform=self.ood_transform)
        rs = torch.utils.data.RandomSampler(testset, True, self.dataset_len)
        ood_iter = DataLoader(testset, batch_size=self.ood_batch_size,
                               shuffle=False, num_workers=12, pin_memory=True, sampler=rs)
        ood_iter.metric_set = 'ood'
        ood_iter.name = self.name
        ood_iter.__len__ = self.dataset_len
        return ood_iter

    def __len__(self):
        return self.dataset_len

class OOD_SVHN(SVHN):
    def __init__(self, root='./data', size=32, batch_sizes={'train':128,'test':100, 'ood':100}, transforms={'train':None, 'test':None, 'ood':None}, dataset_len=2000):
        super(OOD_SVHN, self).__init__(root, size, batch_sizes, transforms)
        self.ood_batch_size = batch_sizes['ood']
        if transforms['ood'] is None:
            self.ood_transform = self.test_transform
        else:
            self.ood_transform = transforms['ood']

        self.dataset_len = dataset_len

    def ood_iter(self):
        testset = dset.SVHN(root=self.root, split='test', download=True, transform=self.ood_transform)
        rs = torch.utils.data.RandomSampler(testset, True, self.dataset_len)
        ood_iter = DataLoader(testset, batch_size=self.ood_batch_size,
                               shuffle=False, num_workers=12, pin_memory=True, sampler=rs)
        ood_iter.metric_set = 'ood'
        ood_iter.name = self.name
        ood_iter.__len__ = self.dataset_len
        return ood_iter

    def __len__(self):
        return self.dataset_len

class OOD_TinyImageNet(TinyImageNet):
    def __init__(self, root='./data', size=32, batch_sizes={'train':128,'test':100, 'ood':100}, transforms={'train':None, 'test':None, 'ood':None}, dataset_len=2000):
        super(OOD_TinyImageNet, self).__init__(root, size, batch_sizes, transforms)
        self.ood_batch_size = batch_sizes['ood']
        if transforms['ood'] is None:
            self.ood_transform = self.test_transform
        else:
            self.ood_transform = transforms['ood']

        self.dataset_len = dataset_len

    def ood_iter(self):
        testset = dset.ImageFolder(root=os.path.join(self.root, 'tiny-imagenet-200/val'), transform=self.ood_transform)
        rs = torch.utils.data.RandomSampler(testset, True, self.dataset_len)
        ood_iter = DataLoader(testset, batch_size=self.ood_batch_size,
                               shuffle=False, num_workers=12, pin_memory=True, sampler=rs)
        ood_iter.metric_set = 'ood'
        ood_iter.name = self.name
        ood_iter.__len__ = self.dataset_len
        return ood_iter

