'''TinyResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from yapwrap.utils import *
from matplotlib import pyplot as plt

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
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
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
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TinyResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(TinyResNet, self).__init__()
        self.name = self.__class__.__name__
        self.in_planes = 64
        self.block = block
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, **kwargs)

        ## Visualization

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def visualize(self, x):

        viz_dict = {}
        out = self.forward(x)
        mhp = HistPlot(title='Model Logit Response',
                                   xlabel='Logit',
                                   ylabel='Frequency',
                                   legend=True,
                                   legend_pos=1,
                                   grid=True)

        mmhp = HistPlot(title='Model Max Logit Response',
                                   xlabel='Logit',
                                   ylabel='Frequency',
                                   legend=True,
                                   legend_pos=1,
                                   grid=True)

        _out = out.detach().cpu().numpy()
        mout = _out.max(1)
        aout = _out.argmax(1)
        for n in range(out.size(1)):
            _x = _out[:,n]
            mhp.add_plot(_x, label=n)
        mmhp.add_plot(mout)
        viz_dict.update({'LogitResponse':torch.from_numpy(mhp.get_image()).permute(2,0,1)})
        viz_dict.update({'MaxLogitResponse':torch.from_numpy(mmhp.get_image()).permute(2,0,1)})
        mhp.close()
        mmhp.close()
        return viz_dict

    def __repr__(self):
        d = {'name':self.name,
             'num_classes':self.num_classes,
             'num_blocks':self.num_blocks}
        return str(d)
    @property
    def default_optimizer_config(self):
        return {"optimizer":{"class":torch.optim.SGD,
                                "params":{"lr":1e-1,
                                          "momentum":0.9,
                                          "nesterov":True,
                                          "weight_decay":5e-4}}}


def TinyResNet18(**kwargs):
    x = TinyResNet(BasicBlock, [2,2,2,2], **kwargs)
    x.name = "{}18".format(x.name)
    return x

def TinyResNet34(**kwargs):
    x = TinyResNet(BasicBlock, [3,4,6,3], **kwargs)
    x.name = "{}34".format(x.name)
    return x

def TinyResNet50(**kwargs):
    x = TinyResNet(Bottleneck, [3,4,6,3], **kwargs)
    x.name = "{}50".format(x.name)
    return x

def TinyResNet101(**kwargs):
    x.name = "{}101".format(x.name)
    return TinyResNet(Bottleneck, [3,4,23,3], **kwargs)

def TinyResNet152(**kwargs):
    x.name = "{}152".format(x.name)
    return TinyResNet(Bottleneck, [3,8,36,3], **kwargs)

