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
    noise = Noise(p,noise_type)

    dataloader.train_transform = tvtfs.Compose([noise, *dataloader.train_transform.transforms])
    dataloader.test_transform = tvtfs.Compose([noise, *dataloader.test_transform.transforms])

    dataloader._class_names = ['noisy_{}'.format(x) for x in dataloader._class_names]
    dataloader.name = 'noise_{}_{}'.format(dataloader.name, str(p).replace('.','_'))
    return dataloader

class Noise(object):
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
