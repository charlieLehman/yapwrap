import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colors
from yapwrap.utils import HistPlot
import numpy as np

__all__ = ['TinyAttention18', 'TinyAttention34', 'TinyAttention50', 'TinyAttention101', 'TinyAttention152']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(x)
        out = F.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(x)
        out = F.relu(self.bn1(self.conv1(out)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return out


class TinyAttention(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(TinyAttention, self).__init__()
        self.name = self.__class__.__name__
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.upsample = lambda x, s: nn.functional.interpolate(x, s, mode='bilinear', align_corners=True)
        self.attn = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.classify = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1, bias=False))
        self.upsample = lambda x, s: nn.functional.interpolate(x, s, mode='bilinear', align_corners=True)
        self.num_classes = num_classes

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def pixelwise_classification(self, x):
        s = (x.size(2), x.size(3))
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.upsample(out, s)
        attn = self.attn(out)
        out = self.classify(out)*attn
        return out, attn

    def visualize(self, x):
        out, attn = self.pixelwise_classification(x)
        segviz = self.overlay_segmentation(x, out)
        x -= x.min()
        x /= x.max()
        viz_dict = {'Input':x, 'Segmentation':segviz, 'Attention':attn}

        mhp = HistPlot(title='Model Logit Response',
                                   xlabel='Logit',
                                   ylabel='Frequency',
                                   legend=True,
                                   legend_pos=1,
                                   grid=True)

        _out = out.sum((-2,-1))/attn.sum((-2,-1))
        for n in range(out.size(1)):
            _x = _out[:,n].detach().cpu().numpy()
            mhp.add_plot(_x, label=n)
        viz_dict.update({'LogitResponse':torch.from_numpy(mhp.get_image()).permute(2,0,1)})
        mhp.close()
        return viz_dict

    def overlay_segmentation(self, x, out):
        conf, pred = F.softmax(out,1).max(1)
        hue = (pred.float() + 0.5)/self.num_classes
        gs_im = x.mean(1)
        gs_im -= gs_im.min()
        gs_im /= gs_im.max()
        hsv_ims = torch.stack((hue, conf, gs_im),-1).cpu().detach().numpy()
        rgb_ims = []
        for x in hsv_ims:
            rgb_ims.append(colors.hsv_to_rgb(x))
        return torch.from_numpy(np.stack(rgb_ims)).permute(0,3,1,2)


    def forward(self, x):
        out, attn = self.pixelwise_classification(x)
        return out.sum((-2,-1))/attn.sum((-2,-1))

class TinySegmentation(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(TinyAttention, self).__init__()
        self.name = self.__class__.__name__
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.upsample = lambda x, s: nn.functional.interpolate(x, s, mode='bilinear', align_corners=True)
        self.classify = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(512, num_classes, kernel_size=1, bias=False))
        self.upsample = lambda x, s: nn.functional.interpolate(x, s, mode='bilinear', align_corners=True)
        self.num_classes = num_classes

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def pixelwise_classification(self, x):
        s = (x.size(2), x.size(3))
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.upsample(out, s)
        out = self.classify(out)
        return out

    def visualize(self, x):
        out = self.pixelwise_classification(x)
        attn = torch.softmax(out, 1).max(1)[0]
        segviz = self.overlay_segmentation(x, out)
        x -= x.min()
        x /= x.max()
        return {'Input':x,
                'Segmentation':segviz,
                'Attention':attn}

    def overlay_segmentation(self, x, out):
        conf, pred = F.softmax(out,1).max(1)
        hue = (pred.float() + 0.5)/self.num_classes
        gs_im = x.mean(1)
        gs_im -= gs_im.min()
        gs_im /= gs_im.max()
        hsv_ims = torch.stack((hue, conf, gs_im),-1).cpu().detach().numpy()
        rgb_ims = []
        for x in hsv_ims:
            rgb_ims.append(colors.hsv_to_rgb(x))
        return torch.from_numpy(np.stack(rgb_ims)).permute(0,3,1,2)


    def forward(self, x):
        out, attn = self.pixelwise_classification(x)
        return out.sum((-2,-1))/attn.sum((-2,-1))

class TinyAttentionDecoder(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(TinyAttentionDecoder, self).__init__()
        self.name = self.__class__.__name__
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 384, num_blocks[3], stride=2)
        self.low_decoder = nn.Sequential(nn.ReLU(),
                                         nn.Conv2d(64,128,1,bias=False),
                                         nn.BatchNorm2d(128))
        self.decoder = nn.Sequential(nn.ReLU(),
                                     nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.Dropout(0.5),
                                     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.Dropout(0.1),
                                     nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self.upsample = lambda x, s: nn.functional.interpolate(x, s, mode='bilinear', align_corners=True)
        self.num_classes = num_classes

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def pixelwise_classification(self, x):
        s = (x.size(2), x.size(3))
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        low = self.low_decoder(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        low = self.upsample(low, s)
        out = self.upsample(out, s)
        out = torch.cat([out,low], 1)
        out = self.decoder(out)

        return out, attn

    def detect_ood(self, x):
        _, attn = self.pixelwise_classification(x)
        return attn.mean((-2,-1))

    def visualize(self, x):
        out, attn = self.pixelwise_classification(x)
        segviz = self.overlay_segmentation(x, out)
        x -= x.min()
        x /= x.max()
        return {'Input':x,
                'Segmentation':segviz,
                'Attention':attn}

    def overlay_segmentation(self, x, out):
        conf, pred = F.softmax(out,1).max(1)
        hue = (pred.float() + 0.5)/self.num_classes
        gs_im = x.mean(1)
        gs_im -= gs_im.min()
        gs_im /= gs_im.max()
        hsv_ims = torch.stack((hue, conf, gs_im),-1).cpu().detach().numpy()
        rgb_ims = []
        for x in hsv_ims:
            rgb_ims.append(colors.hsv_to_rgb(x))
        return torch.from_numpy(np.stack(rgb_ims)).permute(0,3,1,2)

    def forward(self, x):
        out, attn = self.pixelwise_classification(x)
        out = out.sum((-2,-1))/attn.sum((-2,-1))
        return out


def TinyAttention18(**kwargs):
    x = TinyAttention(BasicBlock, [2,2,2,2], **kwargs)
    x.name = "{}18".format(x.name)
    return x

def TinyAttention34(**kwargs):
    x = TinyAttention(BasicBlock, [3,4,6,3], **kwargs)
    x.name = "{}34".format(x.name)
    return x
def TinyAttention50(**kwargs):
    x = TinyAttention(Bottleneck, [3,4,6,3], **kwargs)
    x.name = "{}50".format(x.name)
    return x
def TinyAttention101(**kwargs):
    x = TinyAttention(Bottleneck, [3,4,23,3], **kwargs)
    x.name = "{}101".format(x.name)
    return x
def TinyAttention152(**kwargs):
    x = TinyAttention(Bottleneck, [3,8,36,3], **kwargs)
    x.name = "{}152".format(x.name)
    return x


def TinySegmentation18(**kwargs):
    x = TinySegmentation(BasicBlock, [2,2,2,2], **kwargs)
    x.name = "{}18".format(x.name)
    return x

def TinyAttentionDecoder18(**kwargs):
    x = TinyAttentionDecoder(BasicBlock, [2,2,2,2], **kwargs)
    x.name = "{}18".format(x.name)
    return x


