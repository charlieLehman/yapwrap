import torchvision.transforms as tvtfs
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from yapwrap.dataloaders import Dataloader
import numpy as np
import torch
import os

def to_np(x):
    return x.detach().cpu().numpy()

class ImageFolder(Dataloader):
    def __init__(self, root, size, batch_sizes, transforms={'train':None, 'test':None}):
        super(ImageFolder, self).__init__(root, size, batch_sizes, transforms)
        self.root = root
        self._class_names = os.listdir(os.path.join(self.root, 'train'))
        self.train_batch_size = batch_sizes['train']
        if transforms['train'] is None:
            self.train_transform = tvtfs.Compose([
                tvtfs.RandomResizedCrop(size),
                tvtfs.RandomHorizontalFlip(),
                tvtfs.ToTensor(),
                tvtfs.Normalize((0.485, 0.456, 0.4406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.train_transform = transforms['train']

        self.test_batch_size = batch_sizes['test']
        if transforms['test'] is None:
            self.test_transform = tvtfs.Compose([
                tvtfs.Resize([int(x*1.14) for x in self.size]),
                tvtfs.CenterCrop(self.size),
                tvtfs.ToTensor(),
                tvtfs.Normalize((0.485, 0.456, 0.4406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.test_transform = transforms['test']

    def train_iter(self):
        trainpath = os.path.join(self.root,'train')
        trainset = dset.ImageFolder(root=trainpath, transform=self.train_transform)
        train_iter = DataLoader(trainset, batch_size=self.train_batch_size,
                                shuffle=True, num_workers=12, pin_memory=True)
        train_iter.metric_set = 'train'
        return train_iter

    def test_iter(self):
        return self.val_iter()

    def val_iter(self):
        valpath = os.path.join(self.root,'val')
        valset = dset.ImageFolder(root=valpath, transform=self.test_transform)
        val_iter = DataLoader(valset, batch_size=self.test_batch_size,
                                shuffle=True, num_workers=12, pin_memory=True)
        val_iter.metric_set = 'validation'
        return val_iter

    @property
    def class_names(self):
        return self._class_names

    @property
    def num_classes(self):
        return len(self.class_names)

    @property
    def params(self):
        p = self.param_dict
        p['transforms']['train'] = str(self.train_transform)
        p['transforms']['test'] = str(self.test_transform)
        p.update({'class_names':self.class_names})
        p.update({'num_classes':self.num_classes})
        p.update({'sample_indices':self.example_indices})
        return p

class ImageNet(ImageFolder):
    def __init__(self, root, size=224, batch_sizes={'train':256,'test':100}, transforms={'train':None, 'test':None}):
        super(ImageNet, self).__init__(root, size, batch_sizes, transforms)

