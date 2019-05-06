import torchvision.transforms as tvtfs
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from yapwrap.dataloaders import Dataloader
import numpy as np
import torch
import os
import requests
import tarfile
from google_drive_downloader import GoogleDriveDownloader as gdd



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
                tvtfs.RandomResizedCrop(self.size),
                tvtfs.RandomHorizontalFlip(),
                tvtfs.ToTensor(),
                tvtfs.Normalize((0.485, 0.456, 0.4406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.train_transform = transforms['train']

        self.test_batch_size = batch_sizes['test']
        if transforms['test'] is None:
            self.test_transform = tvtfs.Compose([
                tvtfs.Resize(size),
                tvtfs.CenterCrop(size),
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

class FlickerClean(ImageFolder):
    def __init__(self, root, size=224, batch_sizes={'train':256,'test':100}, transforms={'train':None, 'test':None}):
        self._maybe_download(root)
        super(FlickerClean, self).__init__(root, size, batch_sizes, transforms)

    def _maybe_download(self, root):
        data_url = "1mE24IKvx86QzMoSVeBIM1nT1EP7sLxvm" 
        data_path = os.path.join(root,'flicker_clean_images.tar')
        gdd.download_file_from_google_drive(file_id=data_url,
                                            dest_path=data_path)
        impath = os.path.join(root,'image')
        if not os.path.exists(impath):
            tar = tarfile.open(data_path)
            tar.extractall(root)
            tar.close()
            imlist = []
            for (dirpath, dirnames, filenames) in os.walk(impath):
                imlist.extend(filenames)

            classes = []
            for im in filenames:
                cls = im.split('_')[0]
                imsource = os.path.join(impath,im)
                trn_path = os.path.join(root,'train')
                val_path = os.path.join(root,'val')
                if not os.path.exists(trn_path):
                    os.mkdir(trn_path)
                if not os.path.exists(val_path):
                    os.mkdir(val_path)
                cls_path = os.path.join(trn_path, cls)
                if not os.path.exists(cls_path):
                    os.mkdir(cls_path)
                    os.mkdir(os.path.join(val_path,cls))
                    classes.append(cls)
                imdest = os.path.join(cls_path, im)
                os.rename(imsource,imdest)

            with open(os.path.join(root,'classes.txt'), 'w') as f:
                for it in classes:
                    f.write("%s\n".format(it))

            for cls in classes:
                cls_path = os.path.join(trn_path, cls)
                imlist = []
                for (dirpath, dirnames, filenames) in os.walk(cls_path):
                    imlist.extend(filenames)
                for im in imlist[:100]:
                    imsource = os.path.join(cls_path,im)
                    imdest = os.path.join(val_path,cls,im)
                    os.rename(imsource, imdest)









