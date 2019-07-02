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

model = Runner('PoolNetResNet50_MSRAB', 0)

transform = transforms.Compose([
            transforms.Resize(400),
            transforms.CenterCrop(400),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.4406), (0.229, 0.224, 0.225)),
])
dataset = LIDTrack1Dataset(root='/data/datasets/LID', transform=transform, split='train')
resample = lambda x, s: nn.functional.interpolate(x, scale_factor=s, mode='bilinear', align_corners=True)

for (path, _), (image, label) in tqdm(zip(dataset.example_list, dataset), total=len(dataset)):
    with torch.no_grad():
        image = image.unsqueeze(0)
        output = model(image.cuda()).squeeze()
        hue = (torch.ones_like(output)*label + 0.5)/200
        hsv_im = torch.stack((hue, torch.ones_like(output), output),-1).cpu().detach().numpy()
        col_im = colors.hsv_to_rgb(hsv_im)
        col_im *= 255
        col_im = col_im.astype('uint8')

        image *= torch.tensor((0.229, 0.224, 0.225)).reshape(1,3,1,1)
        image += torch.tensor((0.485, 0.456, 0.4406)).reshape(1,3,1,1)
        image *= 255
        image = image[0].permute(1,2,0).cpu().detach().numpy()
        image = image.astype('uint8')

        imwrite('LID_training_msrab/images/LID_{}.jpg'.format(path), image)
        imwrite('LID_training_msrab/labels/LID_{}.jpg'.format(path), col_im)
