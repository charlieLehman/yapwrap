from __future__ import print_function, division
import os

import numpy as np
import scipy.io
import torch.utils.data as data
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision import transforms
import torch.utils.data as data
from yapwrap.dataloaders import Dataloader
from torch.utils.data import DataLoader
import torch
from yapwrap.dataloaders import *
import yapwrap.dataloaders.yapwrap_transforms as tfs

class_list = ['aeroplane', 'bicycle', 'bird', 'boat','bottle',
              'bus', 'car', 'cat', 'chair','cow', 'diningtable',
              'dog', 'horse','motorbike', 'person', 'pottedplant',
              'sheep', 'sofa', 'train', 'tvmonitor', 'bg']


class SBDSegmentation(Dataloader):
    def __init__(self, root, size=(513,513), batch_sizes={'train':1,'test':4}, transforms={'train':None, 'test':None}):
        self.name = 'SBD'
        self.root = root
        self.size = size
        self.batch_sizes = batch_sizes
        self.transforms = transforms
        self.train_batch_size = batch_sizes['train']

    def train_iter(self):
        trainset = SBDSegmentationDataset(self.root, split='train', **{'batch_sizes':self.batch_sizes, 'transforms':self.transforms})
        train_iter = DataLoader(trainset, batch_size=self.train_batch_size,
                                shuffle=True, num_workers=1, pin_memory=True)
        train_iter.metric_set = 'train'
        return train_iter

    def val_iter(self):
        valset = SBDSegmentationDataset(self.root, split='val', **{'batch_sizes':self.batch_sizes, 'transforms':self.transforms})
        val_iter = DataLoader(valset, batch_size=self.train_batch_size,
                                shuffle=True, num_workers=1, pin_memory=True)
        val_iter.metric_set = 'validation'
        return val_iter

    @property
    def class_names(self):
        return class_list

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

class SBDSegmentationDataset(data.Dataset):
    NUM_CLASSES = 21

    def __init__(self, root, transform, split, size=(513,513), batch_sizes={'train':18,'test':10}, transforms={'train':None, 'test':None}):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = root
        self._dataset_dir = os.path.join(self._base_dir, 'dataset')
        self._image_dir = os.path.join(self._dataset_dir, 'img')
        self._cat_dir = os.path.join(self._dataset_dir, 'cls')
        self.transform = transform


        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split


        # Get list of all images from the split and check that the files exist
        self.im_ids = []
        self.images = []
        self.categories = []
        for splt in self.split:
            with open(os.path.join(self._dataset_dir, splt + '.txt'), "r") as f:
                lines = f.read().splitlines()

            for line in lines:
                _image = os.path.join(self._image_dir, line + ".jpg")
                _categ= os.path.join(self._cat_dir, line + ".mat")
                assert os.path.isfile(_image)
                assert os.path.isfile(_categ)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_categ)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images: {:d}'.format(len(self.images)))


    def __getitem__(self, index):
        _img, _target = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target}
        sample = self.transform(sample)
        return sample['image'], sample['label']

    def __len__(self):
        return len(self.images)

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.fromarray(scipy.io.loadmat(self.categories[index])["GTcls"][0]['Segmentation'][0])

        return _img, _target

    def __str__(self):
        return 'SBDSegmentation(split=' + str(self.split) + ')'

