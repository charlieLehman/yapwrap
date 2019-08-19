import os
import glob
from PIL import Image, ImageMath
import torch
from torch.utils import data
from torchvision import transforms
from yapwrap.dataloaders import yapwrap_transforms as tfs
from yapwrap.dataloaders import Dataloader

class MSRAB(Dataloader):
    def __init__(self, root, size, batch_sizes):
        super(MSRAB, self).__init__(root, size, batch_sizes, transforms=dict())
        self.root = root
        self.train_batch_size = batch_sizes['train']
        self.test_batch_size = batch_sizes['test']
        self.train_transform = transforms.Compose([
            tfs.RandomScaleCrop(base_size=size, crop_size=size, scale=[0.8, 1.2]),
            # tfs.FixScaleCrop(crop_size=size),
            tfs.RandomRotate(15),
            tfs.RandomHorizontalFlip(),
            tfs.RandomGaussianBlur(),
            tfs.Normalize(mean=(0.485, 0.456, 0.4406),
                          std=(0.229, 0.224, 0.225)),
            tfs.ToTensor()
        ])

        self.test_transform= transforms.Compose([
            tfs.FixScaleCrop(crop_size=size),
            tfs.Normalize(mean=(0.485, 0.456, 0.4406),
                          std=(0.229, 0.224, 0.225)),
            tfs.ToTensor()
        ])

    def train_iter(self):
        trainset = MSRABDataset(root=self.root, transform=self.train_transform, split='train')
        train_iter = data.DataLoader(trainset, batch_size=self.train_batch_size,
                                shuffle=True, num_workers=12, pin_memory=True)
        train_iter.metric_set = 'train'
        return train_iter

    def test_iter(self):
        valset = MSRABDataset(root=self.root, transform=self.test_transform, split='test')
        val_iter = data.DataLoader(valset, batch_size=self.test_batch_size,
                                shuffle=False, num_workers=12, pin_memory=True)
        val_iter.metric_set = 'test'
        return val_iter

    def val_iter(self):
        valset = MSRABDataset(root=self.root, transform=self.test_transform, split='valid')
        val_iter = data.DataLoader(valset, batch_size=self.test_batch_size,
                                shuffle=False, num_workers=12, pin_memory=True)
        val_iter.metric_set = 'validation'
        return val_iter

    @property
    def num_classes(self):
        return 1

    @property
    def params(self):
        p = self.param_dict
        p['transforms']['train'] = str(self.train_transform)
        p['transforms']['test'] = str(self.test_transform)
        p.update({'num_classes':self.num_classes})
        return p

class MSRABDataset(data.Dataset):
    """ MSRA-B
    root:    image root (root which contain images)
    """

    def __init__(self, root, transform, split):
        with open(os.path.join(root,'{}.txt'.format(split)), 'r') as f:
            self.label_path = f.readlines()
        self.image_path = list(map(lambda x: x.split('.png')[0]+'.jpg', self.label_path))
        assert len(self.image_path) == len(self.label_path)
        self.transform = transform
        self.root = root

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.root,self.image_path[item]))
        label = Image.open(os.path.join(self.root,self.label_path[item].strip('\n'))).convert('L')
        label = ImageMath.eval('int(a/255)', a=label)
        item = self.transform({'image': image,
                            'label': label})
        return item['image'], item['label'].unsqueeze(0)

    def __len__(self):
        return len(self.image_path)

