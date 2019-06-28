import os
import glob
from PIL import Image, ImageMath
import torch
from torch.utils import data
from torchvision import transforms
from yapwrap.dataloaders import yapwrap_transforms as tfs
from yapwrap.dataloaders import Dataloader

class DUTS(Dataloader):
    def __init__(self, root, size, batch_sizes):
        super(DUTS, self).__init__(root, size, batch_sizes, transforms=dict())
        self.root = root
        self.train_batch_size = batch_sizes['train']
        self.test_batch_size = batch_sizes['test']
        self.train_transform = transforms.Compose([
            tfs.RandomScaleCrop(base_size=size*2, crop_size=size),
            tfs.FixScaleCrop(crop_size=size),
            tfs.RandomHorizontalFlip(),
            tfs.RandomRotate(15),
            tfs.RandomGaussianBlur(),
            tfs.Normalize(mean=(0.4923, 0.4634, 0.3975),
                          std=(0.2275, 0.2246, 0.2302)),
            tfs.ToTensor()
        ])

        self.test_transform= transforms.Compose([
            tfs.FixScaleCrop(crop_size=size),
            tfs.Normalize(mean=(0.4923, 0.4634, 0.3975),
                          std=(0.2275, 0.2246, 0.2302)),
            tfs.ToTensor()
        ])

    def train_iter(self):
        trainset = DUTSDataset(root=self.root, transform=self.train_transform, split='train')
        train_iter = data.DataLoader(trainset, batch_size=self.train_batch_size,
                                shuffle=True, num_workers=12, pin_memory=True)
        train_iter.metric_set = 'train'
        return train_iter

    def test_iter(self):
        valset = DUTSDataset(root=self.root, transform=self.test_transform, split='test')
        val_iter = data.DataLoader(valset, batch_size=self.test_batch_size,
                                shuffle=False, num_workers=12, pin_memory=True)
        val_iter.metric_set = 'test'
        return val_iter

    def val_iter(self):
        valset = DUTSDataset(root=self.root, transform=self.test_transform, split='valid')
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

class DUTSDataset(data.Dataset):
    """ DUTS
    root:    image root (root which contain images)
    """

    def __init__(self, root, transform, split):
        self.split = 'DUTS-TR' if split=='train' else 'DUTS-TE'
        _path = os.path.join(root,self.split)
        self.label_path = os.path.join(_path,'{}-Mask'.format(self.split))
        self.image_path = os.path.join(_path,'{}-Image'.format(self.split))
        self.image_list = [f.split('.png')[0] for f in os.listdir(self.label_path)]
        self.transform = transform
        self.root = root

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.root,self.image_path,self.image_list[item]+'.jpg')).convert('RGB')
        label = Image.open(os.path.join(self.root,self.label_path,self.image_list[item]+'.png')).convert('L')
        label = ImageMath.eval('int(a/255)', a=label)
        item = self.transform({'image': image,
                            'label': label})
        return item['image'], item['label'].unsqueeze(0)

    def __len__(self):
        return len(self.image_list)

