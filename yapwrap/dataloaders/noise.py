import torchvision.transforms as tvtfs
from yapwrap.dataloaders import Dataloader
import numpy as np
from PIL import Image
import torch
import os
import random
from torchvision.transforms.functional import *

def NoisyDataloader(dataloader, p=0.5, noise_type='Gaussian'):
    """
    Noisify a yapwrap dataloader
    """
    noise = AddNoise(p,noise_type)

    dataloader.train_transform = tvtfs.Compose([noise, *dataloader.train_transform.transforms])
    dataloader.test_transform = tvtfs.Compose([noise, *dataloader.test_transform.transforms])

    dataloader._class_names = ['noisy_{}'.format(x) for x in dataloader._class_names]
    dataloader.name = 'noise_{}_{}'.format(dataloader.name, str(p).replace('.','_'))
    return dataloader

class Noise(Dataloader):
    def __init__(self, size=32, batch_sizes={'ood':100}, noise_type='Gaussian', dataset_len=2000):
        super(Noise, self).__init__(None, size, batch_sizes, {'ood':None})
        if isinstance(size, tuple):
            h, w = size
        else:
            h = w = size
        dummy_targets = torch.ones(dataset_len)
        '''
        Original source of noise formulation
        from https://github.com/hendrycks/outlier-exposure
        '''
        if noise_type=='Gaussian':
            noise_data = torch.from_numpy(np.float32(np.clip(
                np.random.normal(size=(dataset_len, 3, h, w), scale=0.5), -1, 1)))
        elif noise_type=='Rademacher':
            noise_data = torch.from_numpy(np.random.binomial(
                n=1, p=0.5, size=(dataset_len, 3, h, w)).astype(np.float32)) * 2 - 1
        elif noise_type=='Blob':
            from skimage.filters import gaussian as gblur
            noise_data = np.float32(np.random.binomial(n=1, p=0.7, size=(dataset_len, h, w, 3)))
            for i in range(dataset_len):
                noise_data[i] = gblur(noise_data[i], sigma=1.5, multichannel=False)
                noise_data[i][noise_data[i] < 0.75] = 0.0

            noise_data = torch.from_numpy(noise_data.transpose((0, 3, 1, 2))) * 2 - 1
        noise_data = torch.utils.data.TensorDataset(noise_data, dummy_targets)
        self.loader = torch.utils.data.DataLoader(noise_data,
                                                batch_size=batch_sizes['ood'],
                                                shuffle=True, num_workers=12,
                                                pin_memory=True)
        self.noise_type = noise_type
        self.name = "{}_{}".format(self.name, noise_type)

    def ood_iter(self):
        ood_iter = self.loader
        ood_iter.metric_set = 'ood'
        ood_iter.name = self.name
        return ood_iter

class AddNoise(object):
    def __init__(self, p=0.5, noise_type='Gaussian'):
        self.p = p

        if noise_type == 'Gaussian':
            self.noise = add_gaussian_noise
        else:
            raise TypeError('{} is not a valid choice of noise_type'.format(noise_type))
    def __call__(self, img):
        return self.noise(img, self.p)

def add_gaussian_noise(img, scale=0.1):
    """Add Gaussian noise to image.

    Args:
        img (PIL Image): Image to be converted to add noise to.

    Returns:
        PIL Image: Noisy version of the image.
    """

    img = np.asarray(img)
    img = img + np.random.randn(*img.shape)*255*scale
    img = np.minimum(np.maximum(img, 0), 255)
    img = img.astype('uint8')
    return Image.fromarray(img)
