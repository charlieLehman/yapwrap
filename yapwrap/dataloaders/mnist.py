import torchvision.transforms as tvtfs
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from yapwrap.dataloaders import Dataloader
import numpy as np
import torch
import os

__all__ = ['MNIST']

def to_np(x):
    return x.detach().cpu().numpy()

class MNIST(Dataloader):
    def __init__(self, root='./data', size=28, batch_sizes={'train':64,'test':64}, transforms={'train':None, 'test':None}):
        super(MNIST, self).__init__(root, size, batch_sizes, transforms)
        self.root = root
        self.train_batch_size = batch_sizes['train']
        if transforms['train'] is None:
            self.train_transform = tvtfs.Compose([
                tvtfs.Resize(self.size),
                tvtfs.RandomCrop(size, padding=4),
                tvtfs.RandomHorizontalFlip(),
                tvtfs.ToTensor(),
                tvtfs.Normalize((0.1307,), (0.3081,)),
            ])
        else:
            self.train_transform = transforms['train']

        self.test_batch_size = batch_sizes['test']
        if transforms['test'] is None:
            self.test_transform = tvtfs.Compose([
                tvtfs.Resize(self.size),
                tvtfs.ToTensor(),
                tvtfs.Normalize((0.1307,), (0.3081,)),
            ])
        else:
            self.test_transform = transforms['test']
        self.example_indices = [4,5,17,47,25,22,3,13,0,27]

    def train_iter(self):
        trainset = dset.MNIST(root=self.root, train=True, download=True, transform=self.train_transform)
        train_iter = DataLoader(trainset, batch_size=self.train_batch_size,
                                shuffle=True, num_workers=12, pin_memory=True)
        train_iter.metric_set = 'train'
        return train_iter

    def test_iter(self):
        testset = dset.MNIST(root=self.root, train=False, download=True, transform=self.test_transform)
        test_iter = DataLoader(testset, batch_size=self.test_batch_size,
                               shuffle=False, num_workers=12, pin_memory=True)
        test_iter.metric_set = 'test'
        return test_iter

    def val_iter(self):
        val_iter = self.test_iter()
        val_iter.metric_set = 'validation'
        return val_iter
    @property
    def class_names(self):
        return ('one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten')

    @property
    def num_classes(self):
        return len(self.class_names)

    @property
    def examples(self):
        path_to_npy = os.path.join(self.root, '{}_examples'.format(self.name))
        try:
            return torch.from_numpy(np.load(path_to_npy + '.npy'))
        except FileNotFoundError:
            test_list = list(self.test_iter())
            x, l = test_list[-1]
            x = to_np(x[self.example_indices])
            np.save(path_to_npy, x)
            return torch.from_numpy(x)

    @property
    def params(self):
        p = self.param_dict
        p['transforms']['train'] = str(self.train_transform)
        p['transforms']['test'] = str(self.test_transform)
        p.update({'class_names':self.class_names})
        p.update({'num_classes':self.num_classes})
        p.update({'sample_indices':self.example_indices})
        return p

