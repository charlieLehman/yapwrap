import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colors
import numpy as np
from yapwrap.utils import *


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class TinyWideAttention(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(TinyWideAttention, self).__init__()
        self.name = self.__class__.__name__
        self.depth = depth
        self.num_classes = num_classes
        self.widen_factor=widen_factor
        self.dropRate=dropRate
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = int((depth - 4) / 6)
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.attn1 = block(nChannels[1], 1, 1)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.attn2 = block(nChannels[2], 1, 1)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        self.attn3 = block(nChannels[3], 1, 1)
        # global average pooling and classifier

        self.classify = nn.Sequential(
            nn.BatchNorm2d(nChannels[3]),
            nn.ReLU(),
            nn.Conv2d(nChannels[3], num_classes, kernel_size=1, stride=1, bias=False)
        )

        self.upsample = lambda x, s: nn.functional.interpolate(x, s, mode='bilinear', align_corners=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def pixelwise_classification(self, x):
        s = (x.size(2), x.size(3))
        out = self.conv1(x)
        out = self.block1(out)
        attn1 = self.attn1(out)
        out = self.block2(out)
        attn2 = self.attn2(out)
        out = self.block3(out)
        attn3 = self.attn3(out)
        out = self.classify(out)
        out = self.upsample(out, s)
        attn = torch.sigmoid(torch.stack([self.upsample(x,s) for x in [attn1, attn2, attn3]],0).sum(0))
        out = out*attn
        return out, attn

    def forward(self, x):
        out, attn = self.pixelwise_classification(x)
        return out.sum((-2,-1))/attn.sum((-2,-1))

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



    def __repr__(self):
        d = {'name':self.name,
             'num_classes':self.num_classes,
             'num_blocks':self.num_blocks}
        return str(d)

    def get_class_params(self):
        modules = [self.block1, self.block2, self.block3, self.conv1, self.classify]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_attn_params(self):
        modules = [self.attn1, self.attn2, self.attn3]
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

class TinyWideSegmentation(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(TinyWideSegmentation, self).__init__()
        self.name = self.__class__.__name__
        self.depth = depth
        self.num_classes = num_classes
        self.widen_factor=widen_factor
        self.dropRate=dropRate
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = int((depth - 4) / 6)
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier

        self.classify = nn.Sequential(
            nn.BatchNorm2d(nChannels[3]),
            nn.ReLU(),
            nn.Conv2d(nChannels[3], num_classes, kernel_size=1, stride=1, bias=False)
        )

        self.upsample = lambda x, s: nn.functional.interpolate(x, s, mode='bilinear', align_corners=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def pixelwise_classification(self, x):
        s = (x.size(2), x.size(3))
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.classify(out)
        return out

    def forward(self, x):
        out = self.pixelwise_classification(x)
        return out.mean((-2,-1))

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


def TinyWideAttention16x8(**kwargs):
    x = TinyWideAttention(depth=16, widen_factor=8, **kwargs)
    x.name = "{}16x8".format(x.name)
    return x

def TinyWideAttention28x10(**kwargs):
    x = TinyWideAttention(depth=28, widen_factor=10, **kwargs)
    x.name = "{}28x10".format(x.name)
    return x

def TinyWideAttention40x2(**kwargs):
    x = TinyWideAttention(depth=40, widen_factor=2, dropRate=0.3, **kwargs)
    x.name = "{}40x2".format(x.name)
    return x

def TinyWideAttention40x14(**kwargs):
    x = TinyWideAttention(BasicBlock, [3,4,6,3], **kwargs)
    x.name = "{}34".format(x.name)
    return x


def TinyWideSegmentation16x8(**kwargs):
    x = TinyWideSegmentation(depth=16, widen_factor=8, **kwargs)
    x.name = "{}16x8".format(x.name)
    return x

def TinyWideSegmentation28x10(**kwargs):
    x = TinyWideSegmentation(depth=28, widen_factor=10, **kwargs)
    x.name = "{}28x10".format(x.name)
    return x

def TinyWideSegmentation40x2(**kwargs):
    x = TinyWideSegmentation(depth=40, widen_factor=2, dropRate=0.3, **kwargs)
    x.name = "{}40x2".format(x.name)
    return x

def TinyWideSegmentation40x14(**kwargs):
    x = TinyWideSegmentation(BasicBlock, [3,4,6,3], **kwargs)
    x.name = "{}34".format(x.name)
    return x

