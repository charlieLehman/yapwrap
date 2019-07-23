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
from PIL import Image, ImageMath

model = Runner('PoolNetResNet50_MSRAB', 0)

transform = transforms.Compose([
            transforms.Resize(400),
            transforms.CenterCrop(400),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.4406), (0.229, 0.224, 0.225)),
])
dataset = LIDTrack1Dataset(root='/data/datasets/LID', transform=transform, split='train')
elist = dataset.example_list
resample = lambda x, s: nn.functional.interpolate(x, scale_factor=s, mode='bilinear', align_corners=True)

for image_name, label in tqdm(elist):
    with torch.no_grad():
            synset, clname = dataset.map_det[label+1]
            folder = image_name.split('_')[0]
            xml_name = os.path.join(dataset.box_path,folder,image_name+'.xml')
            xmin = None
            with open(xml_name) as fp:
                root = ET.fromstring(fp.read())
                for child in root.findall('object'):
                    if child.find('name').text==synset:
                        xmin = int(child.find('bndbox').find('xmin').text)
                        ymin = int(child.find('bndbox').find('ymin').text)
                        xmax = int(child.find('bndbox').find('xmax').text)
                        ymax = int(child.find('bndbox').find('ymax').text)
            image_name  += '.JPEG'
            image = Image.open(os.path.join(dataset.train_path, folder, image_name)).convert('RGB')
            # if xmin is not None:
            image = image.crop((xmin, ymin, xmax, ymax))
            image = transform(image)
        image = image.unsqueeze(0)
        output = model(image.cuda()).squeeze()
        hue = label
        col_im = torch.stack((hue*torch.ones_like(output), torch.ones_like(output)*255, output*255),-1).cpu().detach().numpy()
        col_im = col_im.astype('uint8')

        image *= torch.tensor((0.229, 0.224, 0.225)).reshape(1,3,1,1)
        image += torch.tensor((0.485, 0.456, 0.4406)).reshape(1,3,1,1)
        image *= 255
        image = image[0].permute(1,2,0).cpu().detach().numpy()
        image = image.astype('uint8')

        imwrite('/data/datasets/LID_training_msrab/images/LID_{}.jpg'.format(path), image)
        imwrite('/data/datasets/LID_training_msrab/labels/LID_{}.png'.format(path), col_im)
