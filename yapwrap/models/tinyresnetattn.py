import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colors
from yapwrap.utils import HistPlot
import numpy as np



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
        self.attn1 = block(64, 1, 1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.attn2 = block(128, 1, 1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.attn3 = block(256, 1, 1)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.attn4 = block(512, 1, 1)

        self.classify = nn.Sequential(
            nn.BatchNorm2d(512*block.expansion),
            nn.ReLU(),
            nn.Conv2d(512*block.expansion, num_classes, kernel_size=1, stride=1, bias=False)
        )

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
        attn1 = self.attn1(out)
        out = self.layer2(out)
        attn2 = self.attn2(out)
        out = self.layer3(out)
        attn3 = self.attn3(out)
        out = self.layer4(out)
        attn4 = self.attn4(out)
        out = self.classify(out)
        out = self.upsample(out, s)
        attn = torch.sigmoid(torch.stack([self.upsample(x,s) for x in [attn1, attn2, attn3, attn4]],0).sum(0))
        out = out*attn
        return out, attn

    def visualize(self, x):
        out, attn = self.pixelwise_classification(x)
        smax_attn = torch.softmax(out,1).max(1,keepdim=True)[0]
        segviz = self.overlay_segmentation(x, out)
        x -= x.min()
        x /= x.max()
        viz_dict = {'Input':x, 'Segmentation':segviz, 'Attention':attn, 'SoftMax Attn':smax_attn}

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

        _out = (out.sum((-2,-1))/attn.sum((-2,-1))).detach().cpu().numpy()
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

    def get_class_params(self):
        modules = [self.layer1, self.layer2, self.layer3, self.layer4, self.conv1, self.bn1, self.classify]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_attn_params(self):
        modules = [self.attn1, self.attn2, self.attn3, self.attn4]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def default_optimizer_parameters(self):
        params = [{'params':self.get_class_params(), 'weight_decay':5e-4},
                  {'params':self.get_attn_params(), 'lr':1e-4, 'weight_decay':5e-6}]
        return params

    @property
    def default_optimizer_config(self):
        return {"optimizer":{"class":torch.optim.SGD,
                                "params":{"lr":1e-1,
                                          "momentum":0.9,
                                          "nesterov":True},
                             "optimizer_parameters":str(self.default_optimizer_parameters())}}

class TinySegmentation(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(TinySegmentation, self).__init__()
        self.name = self.__class__.__name__
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.classify = nn.Sequential(nn.ReLU(),
                                      nn.Conv2d(512*block.expansion, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
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
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.classify(out)
        return out

    def visualize(self, x):
        out = self.pixelwise_classification(x)
        attn = torch.softmax(out,1).max(1,keepdim=True)[0]
        s = (x.size(2), x.size(3))
        out = self.upsample(out, s)
        attn = self.upsample(attn, s)
        segviz = self.overlay_segmentation(x, out)
        x -= x.min()
        x /= x.max()
        viz_dict = {'Input':x, 'Segmentation':segviz, 'SoftMax Attention':attn}

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

        _out = out.mean((-2,-1)).detach().cpu().numpy()
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
        out = self.pixelwise_classification(x)
        return out.mean((-2,-1))

    @property
    def default_optimizer_config(self):
        return {"optimizer":{"class":torch.optim.SGD,
                                "params":{"lr":1e-1,
                                          "momentum":0.9,
                                          "nesterov":True,
                                          "weight_decay":5e-4}}}


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

def TinySegmentation34(**kwargs):
    x = TinySegmentation(BasicBlock, [3,4,6,3], **kwargs)
    x.name = "{}34".format(x.name)
    return x
def TinySegmentation50(**kwargs):
    x = TinySegmentation(Bottleneck, [3,4,6,3], **kwargs)
    x.name = "{}50".format(x.name)
    return x
def TinySegmentation101(**kwargs):
    x = TinySegmentation(Bottleneck, [3,4,23,3], **kwargs)
    x.name = "{}101".format(x.name)
    return x
def TinySegmentation152(**kwargs):
    x = TinySegmentation(Bottleneck, [3,8,36,3], **kwargs)
    x.name = "{}152".format(x.name)
    return x


