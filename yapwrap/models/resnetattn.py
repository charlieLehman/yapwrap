import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colors
from matplotlib import pyplot as plt
from yapwrap.utils import HistPlot, GradCAM
from yapwrap.modules import ImplicitComplement, ImplicitAttention
import numpy as np
import math

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


class AttentionBlock(nn.Module):

    def __init__(self, in_planes):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1)

        self.shortcut = nn.Sequential()
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1)
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


class ImpAttn(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ImpAttn, self).__init__()
        self.name = self.__class__.__name__
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.attn1 = AttentionBlock(64*block.expansion)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.attn2 = AttentionBlock(128*block.expansion)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.attn3 = AttentionBlock(256*block.expansion)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.attn4 = AttentionBlock(512*block.expansion)
        self.impattn = ImplicitAttention(1,True)
        self.attn = nn.Sequential(
            AttentionBlock(4),
            nn.Sigmoid(),
            )

        self.classify = nn.Sequential(
            nn.BatchNorm2d(512*block.expansion+64),
            nn.ReLU(),
            nn.Conv2d(512*block.expansion+64, num_classes, kernel_size=1, stride=1, bias=False),
        )

        self.upsample = lambda x, s: nn.functional.interpolate(x, s, mode='bilinear', align_corners=True)
        self.num_classes = num_classes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
        low_level = out
        if self.training:
            attn1 = self.attn1(out)
        out = self.layer2(out)
        if self.training:
            attn2 = self.attn2(out)
        out = self.layer3(out)
        if self.training:
            attn3 = self.attn3(out)
        out = self.layer4(out)
        _s = (out.size(2), out.size(3))
        if self.training:
            attn4 = self.attn4(out)
        low_level = self.upsample(low_level,_s)
        out = torch.cat((out, low_level), dim=1)
        px_log = self.classify(out)
        _impattn = self.impattn(px_log)
        if self.training:
            attn = self.attn(torch.cat([self.upsample(x,_s) for x in [attn1, attn2, attn3, attn4]],1))
        else:
            attn = _impattn
        out = px_log*attn
        pred = out.sum((-2,-1))/attn.sum((-2,-1))
        return out, attn, _impattn, pred

    def visualize(self, x):
        s = (x.size(2), x.size(3))
        out, attn, impattn, pred = self.pixelwise_classification(x)
        out, attn, impattn = self.upsample(out,s), self.upsample(attn,s), self.upsample(_impattn,s)
        smax_attn = torch.softmax(out,1).max(1,keepdim=True)[0]
        segviz = self.overlay_segmentation(x, out)
        x -= x.min()
        x /= x.max()
        viz_dict = {'Input':x,
                    'Segmentation':segviz,
                    'ImplicitAttn':impattn,
                    'Attention':attn,
                    'SoftMax Attention':smax_attn,
        }

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

        response = out.sum((-2,-1))/attn.sum((-2,-1))
        # viz_dict.update({'PxLvlLogits':out, 'PxLvltmax':torch.softmax(out,1)})
        # viz_dict.update({'ImLvlLogits':response, 'ImLvlSoftmax':torch.softmax(response,1)})

        _out = response.detach().cpu().numpy()
        mout = _out.max(1)
        aout = _out.argmax(1)
        for n in range(out.size(1)):
            _x = _out[:,n]
            mhp.add_plot(_x, label=n)
        mmhp.add_plot(mout)
        mhp.close()
        mmhp.close()

        return viz_dict

    def overlay_segmentation(self, x, out):
        conf, pred = F.softmax(out,1).max(1)
        hue = (pred.float() + 0.5)/self.num_classes
        if self.num_classes == 10:
            hd = torch.linspace(.05,.95,10)
            hd = hd[[0,2,4,6,8,3,7,5,1,9]].to(out.device)
            # hd = torch.from_numpy(colors.rgb_to_hsv(plt.get_cmap('tab10').colors)[:,0]).float()
            # hd = hd.to(out.device)
            # # y_1h = y[pred].permute(0,3,1,2).to(out.device)
            hue = hd[pred]
        gs_im = x.mean(1)
        gs_im -= gs_im.min()
        gs_im /= gs_im.max()
        hsv_ims = torch.stack((hue, conf, gs_im),-1).cpu().detach().numpy()
        rgb_ims = []
        for x in hsv_ims:
            rgb_ims.append(colors.hsv_to_rgb(x))
        return torch.from_numpy(np.stack(rgb_ims)).permute(0,3,1,2)

    def forward(self, x):
        return self.pixelwise_classification(x)

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
        params = [{'params':self.get_class_params(), 'weight_decay':1e-4},
                  {'params':self.get_attn_params(), 'lr':1e-1, 'weight_decay':1e-4}]
        return params

    @property
    def default_optimizer_config(self):
        return {"optimizer":{"class":torch.optim.SGD,
                                "params":{"lr":1e-1,
                                          "momentum":0.9,
                                          "nesterov":True},
                             "optimizer_parameters":str(self.default_optimizer_parameters())}}
                             # "optimizer_parameters":self.parameters}}


def ImpAttn18(**kwargs):
    x = ImpAttn(BasicBlock, [2,2,2,2], **kwargs)
    x.name = "{}18".format(x.name)
    return x

def ImpAttn34(**kwargs):
    x = ImpAttn(BasicBlock, [3,4,6,3], **kwargs)
    x.name = "{}34".format(x.name)
    return x
def ImpAttn50(**kwargs):
    x = ImpAttn(Bottleneck, [3,4,6,3], **kwargs)
    x.name = "{}50".format(x.name)
    return x
def ImpAttn101(**kwargs):
    x = ImpAttn(Bottleneck, [3,4,23,3], **kwargs)
    x.name = "{}101".format(x.name)
    return x
def ImpAttn152(**kwargs):
    x = ImpAttn(Bottleneck, [3,8,36,3], **kwargs)
    x.name = "{}152".format(x.name)
    return x


