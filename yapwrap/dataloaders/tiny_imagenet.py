import torchvision.transforms as tvtfs
import torchvision.datasets as dset
from torchvision.datasets.utils import download_url
import zipfile
from torch.utils.data import DataLoader
from yapwrap.dataloaders import Dataloader
import numpy as np
import torch
import os

__all__ = ['TinyImageNet']

def to_np(x):
    return x.detach().cpu().numpy()

class TinyImageNet(Dataloader):
    def __init__(self, root='./data', size=64, batch_sizes={'train':128,'test':100}, transforms={'train':None, 'test':None}):
        super(TinyImageNet, self).__init__(root, size, batch_sizes, transforms)
        self.root = root
        self.train_batch_size = batch_sizes['train']
        self.path = os.path.join(self.root, 'tiny-imagenet-200')
        if not os.path.exists(self.path):
            download_url('http://cs231n.stanford.edu/tiny-imagenet-200.zip', self.root, 'tiny-imagenet-200.zip', None)
            with zipfile.ZipFile('{}.zip'.format(self.path), 'r') as z:
                z.extractall(self.root)
        if os.path.exists(os.path.join(self.path,'val/images')):
            self.create_val_folder()

        if transforms['train'] is None:
            self.train_transform = tvtfs.Compose([
                tvtfs.Resize(self.size),
                tvtfs.RandomHorizontalFlip(),
                tvtfs.RandomCrop(self.size, padding=4),
                tvtfs.ToTensor(),
                tvtfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.train_transform = transforms['train']

        self.test_batch_size = batch_sizes['test']
        if transforms['test'] is None:
            self.test_transform = tvtfs.Compose([
                tvtfs.Resize(self.size),
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
        valset = dset.ImageFolder(root=os.path.join(self.root, 'tiny-imagenet-200/val'), transform=self.test_transform)
        val_iter = DataLoader(valset, batch_size=self.test_batch_size, shuffle=True, num_workers=12, pin_memory=True)
        val_iter.metric_set = 'validation'
        return val_iter

    def build_label_dicts(self):
        label_dict, class_description = {}, {}
        with open(os.path.join(self.root, 'tiny-imagenet-200/wnids.txt'), 'r') as f:
            for i, line in enumerate(f.readlines()):
                synset = line[:-1]  # remove \n
                label_dict[synset] = i
        with open(os.path.join(self.root, 'tiny-imagenet-200/words.txt'), 'r') as f:
            for i, line in enumerate(f.readlines()):
                synset, desc = line.split('\t')
                desc = desc[:-1]  # remove \n
                if synset in label_dict:
                    class_description[label_dict[synset]] = desc
        return label_dict, class_description

    @property
    def class_names(self):
        return self._class_names

    @property
    def num_classes(self):
        return 200

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

    def create_val_folder(self):
        """
        This method is responsible for separating validation images into separate sub folders
        """
        print('Creating val set folder structure')
        path = os.path.join(self.root, 'tiny-imagenet-200/val')
        filename = os.path.join(self.root, 'tiny-imagenet-200/val/val_annotations.txt')
        fp = open(filename, "r")  # open file in read mode
        data = fp.readlines()  # read line by line

        val_img_dict = {}
        for line in data:
            words = line.split("\t")
            val_img_dict[words[0]] = words[1]
        fp.close()

        # Create folder if not present, and move image into proper folder
        for img, folder in val_img_dict.items():
            newpath = (os.path.join(path, folder))
            if not os.path.exists(newpath):  # check if folder exists
                os.makedirs(newpath)

            if os.path.exists(os.path.join(path,'images', img)):  # Check if image exists in default directory
                os.rename(os.path.join(path,'images', img), os.path.join(newpath, img))
        os.rmdir(os.path.join(path,'images'))
