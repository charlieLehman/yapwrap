import torchvision.transforms as tvtfs
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from yapwrap.dataloaders import *
import numpy as np
import torch
import os

class OODDataloaders(object):
    def __init__(self, dataloader_list):
        for dataloader in dataloader_list:
            if not isinstance(dataloader, Dataloader):
                raise TypeError('{} is not a valid type yapwrap.Dataloader'.format(type(dataloader).__name__))
            if not 'OOD' in type(dataloader).__name__ and not 'Noise' in type(dataloader).__name__:
                raise TypeError('{} is not a valid OOD or Noise Dataset'.format(type(dataloader).__name__))
        self.ood_dataloaders = dataloader_list

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            dataloader: (Dataloader)
        """

        return self.ood_dataloaders[index]

    def __len__(self):
        return len(self.ood_dataloaders)

    def __repr__(self):
        return [c.name for c in self.ood_dataloaders]


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
