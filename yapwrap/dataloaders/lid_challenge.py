import os
import glob
from PIL import Image, ImageMath
import torch
from torch.utils import data
from torchvision import transforms
from yapwrap.dataloaders import yapwrap_transforms as tfs
from yapwrap.dataloaders import Dataloader

class LIDTrack1(Dataloader):
    def __init__(self, root, size, batch_sizes):
        super(LIDTrack1, self).__init__(root, size, batch_sizes, transforms=dict())
        self.root = root
        self.train_batch_size = batch_sizes['train']
        self.val_batch_size = batch_sizes['val']
        self.test_batch_size = batch_sizes['test']
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.4406), (0.229, 0.224, 0.225)),
        ])

        self.test_transform = transforms.Compose([
            tfs.FixScaleCrop(crop_size=size),
            tfs.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tfs.ToTensor()])

    def train_iter(self):
        trainset = LIDTrack1Dataset(root=self.root, transform=self.train_transform, split='train')
        train_iter = data.DataLoader(trainset, batch_size=self.train_batch_size,
                                shuffle=True, num_workers=12, pin_memory=True)
        train_iter.metric_set = 'train'
        return train_iter

    def test_iter(self):
        testset = LIDTrack1Dataset(root=self.root, transform=self.test_transform, split='test')
        test_iter = data.DataLoader(valset, batch_size=self.test_batch_size,
                                shuffle=False, num_workers=12, pin_memory=True)
        test_iter.metric_set = 'test'
        return test_iter

    def val_iter(self):
        valset = LIDTrack1Dataset(root=self.root, transform=self.test_transform, split='val')
        val_iter = data.DataLoader(valset, batch_size=self.val_batch_size,
                                shuffle=False, num_workers=12, pin_memory=True)
        val_iter.metric_set = 'validation'
        return val_iter

    @property
    def num_classes(self):
        return 200

    @property
    def params(self):
        p = self.param_dict
        p['transforms']['train'] = str(self.train_transform)
        p['transforms']['test'] = str(self.test_transform)
        p.update({'num_classes':self.num_classes})
        return p

class LIDTrack1Dataset(data.Dataset):
    """ LID Track 1 Dataset
    root: location of LID folder
    Folder structure:

    LID
    |- LID_track1
    |- ILSVRC2013_devkit
    |- ILSVRC
    """

    def __init__(self, root, transform, split):
        self.root = root
        self.transform = transform
        self.num_classes = 200
        self.split = split
        self.det_list_path = os.path.join(root,"ILSVRC2013_devkit/data/det_lists")
        if split=='train':
            self.train_path = os.path.join(root,"ILSVRC/Data/DET/train/ILSVRC2013_train")
            self.pos_det_list = glob.glob("{}/*_pos_*txt".format(self.det_list_path))
            self.example_list = []
            for c in self.pos_det_list:
                with open(c, 'r') as f:
                    class_num = int(c.split('.')[0].split('_')[-1])
                    for im_path in f.readlines():
                        self.example_list.append((im_path.strip('\n'), class_num-1))
        elif split=='val':
            self.example_list = []
            with open(os.path.join(root, "LID_track1","val.txt"), 'r') as f:
                for im_path in f.readlines():
                    self.example_list.append(im_path.strip('\n'))

    def __getitem__(self, item):
        if self.split=='train':
            image_name, label = self.example_list[item]
            image_name  += '.JPEG'
            folder = image_name.split('_')[0]
            image = Image.open(os.path.join(self.train_path, folder,image_name)).convert('RGB')
            image = self.transform(image)
            return image, label
        elif self.split=='val':
            example_name = self.example_list[item]
            image = Image.open(os.path.join(self.root,'LID_track1', self.split,example_name+'.JPEG')).convert('RGB')
            label = Image.open(os.path.join(self.root,'track1_val_annotations_raw', example_name+'.png')).convert('L')
            example = self.transform({'image': image, 'label': label})
            return example['image'], example['label'].unsqueeze(0)
        else:
            example_name = self.example_list[item]
            image = Image.open(os.path.join(self.root,'LID_track1', self.split,example_name+'.JPEG')).convert('RGB')
            label = Image.new('L', image.size)
            example = self.transform({'image': image,
                                'label': label})
            return example['image'], example['label'].unsqueeze(0)

    def __len__(self):
        return len(self.example_list)

