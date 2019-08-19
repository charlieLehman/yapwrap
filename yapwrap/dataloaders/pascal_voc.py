from __future__ import print_function, division
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

# __all__ = ['VOCSegmentation']

class VOCSegmentation(Dataloader):
    def __init__(self, root, size=(513,513), batch_sizes={'train':1,'test':4}, transforms={'train':None, 'test':None}):
        self.name = 'PascalVOC'
        self.root = root
        self.size = size
        self.batch_sizes = batch_sizes
        self.transforms = transforms
        self.train_batch_size = batch_sizes['train']

    def train_iter(self):
        trainset = VOCSegmentationDataset(self.root, split='train', **{'batch_sizes':self.batch_sizes, 'transforms':self.transforms})
        train_iter = DataLoader(trainset, batch_size=self.train_batch_size,
                                shuffle=True, num_workers=1, pin_memory=True)
        train_iter.metric_set = 'train'
        return train_iter

    def val_iter(self):
        valset = VOCSegmentationDataset(self.root, split='val', **{'batch_sizes':self.batch_sizes, 'transforms':self.transforms})
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


class VOCSegmentationDataset(data.Dataset):
    """
    PascalVoc dataset
    """
    NUM_CLASSES = 21
    def __init__(self, root, split, transform, size=(513,513), batch_sizes={'train':18,'test':10}):
        """
        :param root: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = root
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')
        self.split = [split]
        self.args = {'base_size':513, 'crop_size': 513}
        self.transform = transform

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                assert os.path.isfile(_image)
                if self.split[0] != 'test':
                    _cat = os.path.join(self._cat_dir, line + ".png")
                    if os.path.isfile(_cat):
                        self.categories.append(_cat)
                self.im_ids.append(line)
                self.images.append(_image)

        if self.split[0] != 'test':
            assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _sample = self._make_img_gt_point_pair(index)
        if self.split[0] == 'test':
            _img, _target, _imname = _sample
            _imname = _imname.split('/')[-1].split('.')[0]

        else:
            _img, _target = _sample

        sample = {'image': _img, 'label': _target}
        sample = self.transform(sample)

        return sample['image'], sample['label'].long()

    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        if self.split[0] == 'test':
            _target = Image.new('L', _img.size, color=(0))
            return _img, _target, self.images[index]
        else:
            _target = Image.open(self.categories[index])
        return _img, _target

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'


