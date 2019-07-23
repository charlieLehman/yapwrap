from yapwrap.experiments import *
from yapwrap.dataloaders import *
from yapwrap.models import *
from yapwrap.modules import *
from yapwrap.evaluators import *
from yapwrap.loggers import *
from torchvision import transforms
from yapwrap.dataloaders import yapwrap_transforms as tfs
import torch
import torch.nn as nn
from imageio import imwrite
from matplotlib import colors
from tqdm import tqdm
import itertools
import xml.etree.ElementTree as ET
from multiprocessing import Pool

transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.4406), (0.229, 0.224, 0.225)),
])
split = 'val'

dataset = LIDTrack1Dataset(root='/data/datasets/LID', transform=transform, split=split)
elist = dataset.example_list
resample = lambda x, s: nn.functional.interpolate(x, scale_factor=s, mode='bilinear', align_corners=True)
for label in range(200):
    synset, clname = dataset.map_det[label+1]
    fold_path = '/data/datasets/LID_training_imageclassification/{}/{:03d}_{}'.format(split,label, clname.replace('\n',''))
    if not os.path.isdir(fold_path):
        os.mkdir(fold_path)

def make_(x):
    image_name, label = x
    synset, clname = dataset.map_det[label+1]
    folder = image_name.split('_')[0]
    xml_name = os.path.join(dataset.box_path,folder,image_name+'.xml')
    xmin = None
    with open(xml_name) as fp:
        root = ET.fromstring(fp.read())
        image_name  += '.JPEG'
        image_ = Image.open(os.path.join(dataset.train_path, folder, image_name)).convert('RGB')
        for child in root.findall('object'):
            if child.find('name').text==synset:
                xmin = int(child.find('bndbox').find('xmin').text)
                ymin = int(child.find('bndbox').find('ymin').text)
                xmax = int(child.find('bndbox').find('xmax').text)
                ymax = int(child.find('bndbox').find('ymax').text)
                # if xmin is not None:
                x_d = xmax-xmin
                y_d = ymax-ymin
                if x_d > 25 and y_d > 25:
                    image = image_.crop((xmin, ymin, xmax, ymax))
                    folder = image_name.split('_')[0]
                    fold_path = '/data/datasets/LID_training_imageclassification/{}/{:03d}_{}'.format(split,label, clname.replace('\n',''))
                    image.save(os.path.join(fold_path,'LID_{}'.format(image_name)), "JPEG")

if __name__=='__main__':
    p = Pool(8)
    for _ in tqdm(p.imap_unordered(make_,elist), total=len(elist)):
        pass
