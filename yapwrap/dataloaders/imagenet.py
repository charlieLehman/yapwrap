import torchvision.transforms as tvtfs
import torchvision.datasets as dset
from torchvision.datasets.utils import download_url
import zipfile
from torch.utils.data import DataLoader
from yapwrap.dataloaders import Dataloader
import numpy as np
import torch
import os

__all__ = ['ImageNet']

def to_np(x):
    return x.detach().cpu().numpy()

class ImageNet(Dataloader):
    def __init__(self, root='./data', size=224, batch_sizes={'train':32,'test':20}, transforms={'train':None, 'test':None}):
        super(ImageNet, self).__init__(root, size, batch_sizes, transforms)
        self.root = root
        self.train_batch_size = batch_sizes['train']
        self.path = os.path.join(self.root, 'tiny-imagenet-200')
        if not os.path.exists(self.path):
            download_url('http://cs231n.stanford.edu/tiny-imagenet-200.zip', self.root, 'tiny-imagenet-200.zip', None)
            with zipfile.ZipFile('{}.zip'.format(self.path), 'r') as z:
                z.extractall(self.root)

        if transforms['train'] is None:
            self.train_transform = tvtfs.Compose([
                tvtfs.RandomResizedCrop(self.size),
                tvtfs.RandomHorizontalFlip(),
                tvtfs.ToTensor(),
                tvtfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.train_transform = transforms['train']

        self.test_batch_size = batch_sizes['test']
        if transforms['test'] is None:
            self.test_transform = tvtfs.Compose([
                tvtfs.Resize(self.size+32),
                tvtfs.CenterCrop(self.size),
                tvtfs.ToTensor(),
                tvtfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.test_transform = transforms['test']
        self.example_indices = None
        self._class_names = None

    def train_iter(self):
        trainset = dset.ImageFolder(root=os.path.join(self.root, 'tiny-imagenet-200/train'), transform=self.train_transform)
        train_iter = DataLoader(trainset, batch_size=self.train_batch_size, shuffle=True, num_workers=12, pin_memory=True)

        train_iter.metric_set = 'train'
        return train_iter

    def test_iter(self):
        testset = dset.ImageFolder(root=os.path.join(self.root, 'tiny-imagenet-200/val'), transform=self.test_transform)
        test_iter = DataLoader(testset, batch_size=self.test_batch_size, shuffle=False, num_workers=12, pin_memory=True)
        test_iter.metric_set = 'test'
        return test_iter

    def val_iter(self):
        val_iter = self.test_iter()
        val_iter.metric_set = 'validation'
        return val_iter

    @property
    def class_names(self):
        return self._class_names

    @property
    def num_classes(self):
        return 1000

    @property
    def examples(self):
        return False

    @property
    def params(self):
        p = self.param_dict
        p['transforms']['train'] = str(self.train_transform)
        p['transforms']['test'] = str(self.test_transform)
        p.update({'class_names':self.class_names})
        p.update({'num_classes':self.num_classes})
        p.update({'sample_indices':self.example_indices})
        return p
