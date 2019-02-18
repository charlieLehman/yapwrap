import torchvision.transforms as tvtfs
import torchvision.datasets as dset
from torch.utils.data import DataLoader
import numpy as np
import torch
import os

__all__ = ['Dataloader','CIFAR10', 'MNIST', 'FASHION_MNIST', 'SVHN']

def to_np(x):
    return x.detach().cpu().numpy()

class Dataloader(object):
    def __init__(self, root, size, batch_sizes, transforms):

        self.name = self.__class__.__name__

        if not isinstance(root, str):
            raise TypeError('{} is not a valid type str'.format(type(root).__name__))
        if not os.path.exists(root):
            print('creating {}'.format(root))
            os.mkdir(root)
        self.root = root
        if isinstance(size, tuple):
            self.size = size
        elif isinstance(size, int):
            self.size = (size, size)
        else:
            raise TypeError('{} is not a valid type (int,int) or int'.format(type(size).__name__))
        if not isinstance(batch_sizes, dict):
            raise TypeError('{} is not a valid type with format {\'train\':int,\'test\':int, \'val\':int,...etc.}'.format(type(batch_sizes).__name__))

        if not isinstance(transforms, dict):
            raise TypeError('{} is not a valid type with format {\'train\':transform,\'test\':transform, \'val\':transform,...etc.}'.format(type(transforms).__name__))

        for v in transforms.values():
            if not callable(v) and v is not None:
                raise TypeError('{} is not a valid transform'.format(type(v).__name__))
        str_transforms = {}
        for i,v in transforms.items():
            try:
                vname = v.__class__.__name__
            except:
                vname = v.__name__
            str_transforms.update({i:vname})

        self.param_dict = {'root':root,
                      'size':size,
                      'batch_sizes':batch_sizes,
                      'transforms':str_transforms}

    def train_iter(self):
        raise NotImplementedError

    def test_iter(self):
        raise NotImplementedError

    @property
    def class_names(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        raise NotImplementedError

    @property
    def examples(self):
        raise NotImplementedError
    @property
    def params(self):
        return self.param_dict
    def __repr__(self):
        return str(self.params)

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

class FASHION_MNIST(Dataloader):
    def __init__(self, root='./data', size=28, batch_sizes={'train':64,'test':64}, transforms={'train':None, 'test':None}):
        super(FASHION_MNIST, self).__init__(root, size, batch_sizes, transforms)
        self.root = root
        self.train_batch_size = batch_sizes['train']
        if transforms['train'] is None:
            self.train_transform = tvtfs.Compose([
                tvtfs.Resize(self.size),
                tvtfs.RandomCrop(size, padding=4),
                tvtfs.RandomHorizontalFlip(),
                tvtfs.ToTensor(),
            ])
        else:
            self.train_transform = transforms['train']

        self.test_batch_size = batch_sizes['test']
        if transforms['test'] is None:
            self.test_transform = tvtfs.Compose([
                tvtfs.Resize(self.size),
                tvtfs.ToTensor(),
            ])
        else:
            self.test_transform = transforms['test']
        self.example_indices = [4,5,17,47,25,22,3,13,0,27]

    def train_iter(self):
        trainset = dset.FashionMNIST(root=self.root, train=True, download=True, transform=self.train_transform)
        train_iter = DataLoader(trainset, batch_size=self.train_batch_size,
                                shuffle=True, num_workers=12, pin_memory=True)
        train_iter.metric_set = 'train'
        return train_iter

    def test_iter(self):
        testset = dset.FashionMNIST(root=self.root, train=False, download=True, transform=self.test_transform)
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
        return ('T-Shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

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


class SVHN(Dataloader):
    def __init__(self, root='./data', size=32, batch_sizes={'train':128,'test':100}, transforms={'train':None, 'test':None}):
        super(SVHN, self).__init__(root, size, batch_sizes, transforms)
        self.root = root
        self.train_batch_size = batch_sizes['train']
        if transforms['train'] is None:
            self.train_transform = tvtfs.Compose([
                tvtfs.Resize(self.size),
                tvtfs.RandomCrop(size, padding=4),
                tvtfs.RandomHorizontalFlip(),
                tvtfs.ToTensor(),
            ])
        else:
            self.train_transform = transforms['train']

        self.test_batch_size = batch_sizes['test']
        if transforms['test'] is None:
            self.test_transform = tvtfs.Compose([
                tvtfs.Resize(self.size),
                tvtfs.ToTensor(),
            ])
        else:
            self.test_transform = transforms['test']
        self.example_indices = [4,5,17,47,25,22,3,13,0,27]

    def train_iter(self):
        trainset = dset.SVHN(root=self.root, split='train', download=True, transform=self.train_transform)
        train_iter = DataLoader(trainset, batch_size=self.train_batch_size,
                                shuffle=True, num_workers=12, pin_memory=True)
        train_iter.metric_set = 'train'
        return train_iter

    def test_iter(self):
        testset = dset.SVHN(root=self.root, split='test', download=True, transform=self.test_transform)
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
        return ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')

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

class CIFAR10(Dataloader):
    def __init__(self, root='./data', size=32, batch_sizes={'train':128,'test':100}, transforms={'train':None, 'test':None}):
        super(CIFAR10, self).__init__(root, size, batch_sizes, transforms)
        self.root = root
        self.train_batch_size = batch_sizes['train']
        if transforms['train'] is None:
            self.train_transform = tvtfs.Compose([
                tvtfs.Resize(self.size),
                tvtfs.RandomCrop(size, padding=4),
                tvtfs.RandomHorizontalFlip(),
                tvtfs.RandomAffine(25,(.1,.1),(.9,1.1), resample=3),
                tvtfs.ToTensor(),
                tvtfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.train_transform = transforms['train']

        self.test_batch_size = batch_sizes['test']
        if transforms['test'] is None:
            self.test_transform = tvtfs.Compose([
                tvtfs.Resize(self.size),
                tvtfs.ToTensor(),
                tvtfs.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            self.test_transform = transforms['test']
        self.example_indices = [4,5,17,47,25,22,3,13,0,27]

    def train_iter(self):
        trainset = dset.CIFAR10(root=self.root, train=True, download=True, transform=self.train_transform)
        train_iter = DataLoader(trainset, batch_size=self.train_batch_size,
                                shuffle=True, num_workers=12, pin_memory=True)
        train_iter.metric_set = 'train'
        return train_iter

    def test_iter(self):
        testset = dset.CIFAR10(root=self.root, train=False, download=True, transform=self.test_transform)
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
        return ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

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
