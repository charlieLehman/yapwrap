import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from matplotlib import colors
from matplotlib import pyplot as plt
from yapwrap.utils import HistPlot, GradCAM
from yapwrap.modules import ImplicitComplement, ImplicitAttention
from .poolnet import PoolNetResNet50
import numpy as np
import math

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
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

class ASPPBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dilations, kernel_size=3):
        super(ASPPBlock, self).__init__()

        _at_conv = lambda p: nn.Sequential(
            nn.Conv2d(in_planes, in_planes*2, kernel_size=3 if p!=1 else 1, stride=1, padding=0 if p==1 else p, dilation=p, bias=False),
            nn.BatchNorm2d(in_planes*2),
        )
        self.aspp = nn.ModuleList([_at_conv(p) for p in dilations])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(in_planes, in_planes*2, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(in_planes*2))
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_planes*2*(len(dilations)+1), out_planes, kernel_size=1, padding=0),
        )

    def forward(self, x):
        _s = x.size()[2:]
        out = F.relu(x)
        outs = []
        for _atrous_conv in self.aspp:
            y = _atrous_conv(out)
            outs.append(y)
        outs.append(F.interpolate(self.global_avg_pool(out), _s, mode='bilinear', align_corners=True))
        out = torch.cat(outs, dim=1)
        out = self.conv(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, in_planes, dilations):
        super(AttentionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_planes, 256, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=5, padding=2, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=7, padding=3, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1, padding=0, stride=1, bias=False),
        )
        # self.conv = ASPPBlock(in_planes, 1, dilations)
        self.upsample = lambda x, s: nn.functional.interpolate(x, s, mode='bilinear', align_corners=True)

    def forward(self, x):
        s = (x.size(2), x.size(3))
        out = self.upsample(x,16)
        out = self.conv(x)
        out = self.upsample(out,s)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        self.dilation=dilation
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
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
    def __init__(self, block, num_blocks, pretrained_attn_path=None, num_classes=10, tiny=False, optimizer_config=None, pretrained=True):
        super(ImpAttn, self).__init__()
        self.name = self.__class__.__name__
        self.in_planes = 64

        if tiny:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.bn1 = nn.BatchNorm2d(64)
        self.tiny = tiny
        dilations = [1,2,4] if self.tiny else [1,6,12,18]

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dilation=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dilation=1)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dilation=1)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=1, dilation=2)
        self.impattn = ImplicitAttention(1,True)

        self.aspp = ASPPBlock(512*block.expansion,256,dilations)
        self.classify = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(320, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, stride=1, bias=False),
        )

        self.upsample = lambda x, s: nn.functional.interpolate(x, s, mode='bilinear', align_corners=True)
        self.num_classes = num_classes

        if pretrained:
            self._load_pretrained_model()
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        self.attn = PoolNetResNet50()
        state = torch.load(pretrained_attn_path)
        self.attn.load_state_dict(state['model_state_dict'])
        for param in self.attn.parameters():
            param.requires_grad = False

        self.initialize_optimization_parameters(optimizer_config)

    def _make_layer(self, block, planes, num_blocks, stride, dilation):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dilation))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def pixelwise_classification(self, x):
        s = (x.size(2), x.size(3))
        if self.training:
            attn = self.attn(x)
        out = self.bn1(self.conv1(x))
        _s = (out.size(2), out.size(3))
        low_level = out
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.aspp(out)
        out = self.upsample(out, _s)
        out = torch.cat((out, low_level), dim=1)
        px_log = self.classify(out)
        _impattn = self.impattn(px_log)
        if not self.training:
            attn = _impattn
        attn = self.upsample(attn, _s)
        out = px_log*attn
        pred = out.sum((-2,-1))/attn.sum((-2,-1))
        return out, attn, _impattn, pred

    def forward(self, x):
        return self.pixelwise_classification(x)

    def visualize(self, *input):
        x, target = input
        out, attn, impattn, pred = self.pixelwise_classification(x)
        s = (x.size(2), x.size(3))
        out, attn, impattn = self.upsample(out,s), self.upsample(attn,s), self.upsample(impattn,s)
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

    def overlay_segmentation(self, x, out, conf=None):
        if conf is None:
            conf, pred = torch.softmax(out,1).max(1)
        else:
            pred = out.argmax(1)
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

    def get_class_params(self):
        modules = [self.layer1, self.layer2, self.layer3, self.layer4, self.conv1, self.bn1]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_params(self):
        modules = [self.classify, self.aspp]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def initialize_optimization_parameters(self, optimizer_config):
        cp = dict(optimizer_config['class_params'])
        ap = dict(optimizer_config['10x_params'])
        cp['params'] = self.get_class_params()
        ap['params'] = self.get_10x_params()
        self._optimizer_parameters = [cp, ap]
        self._default_optimizer_config = dict(optimizer_config['optimizer'])

    def optimizer_parameters(self):
        return self._optimizer_parameters

    @property
    def optimizer_config(self):
        return self._default_optimizer_config

    def _load_pretrained_model(self):
        # pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet18-5c106cde.pth')
        print("Loading ResNet50")
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


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

def TinyImpAttn18(**kwargs):
    x = ImpAttn(BasicBlock, [2,2,2,2], tiny=True, **kwargs)
    x.name = "Tiny{}18".format(x.name)
    return x

def TinyImpAttn34(**kwargs):

    x = ImpAttn(BasicBlock, [3,4,6,3], tiny=True, **kwargs)
    x.name = "Tiny{}34".format(x.name)
    return x
def TinyImpAttn50(**kwargs):
    x = ImpAttn(Bottleneck, [3,4,6,3], tiny=True, **kwargs)
    x.name = "Tiny{}50".format(x.name)
    return x
def TinyImpAttn101(**kwargs):
    x = ImpAttn(Bottleneck, [3,4,23,3], tiny=True, **kwargs)
    x.name = "Tiny{}101".format(x.name)
    return x
def TinyImpAttn152(**kwargs):
    x = ImpAttn(Bottleneck, [3,8,36,3], tiny=True,**kwargs)
    x.name = "Tiny{}152".format(x.name)
    return x


