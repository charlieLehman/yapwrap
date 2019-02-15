import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ResNet18ImpBG','ResNet18ImpBGDecoder',]


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


class ResNetImpBG(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetImpBG, self).__init__()
        self.name = self.__class__.__name__
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, num_classes, num_blocks[3], stride=2)
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

        # Extract the negative max complement logits
        c_out = torch.zeros_like(out)
        for k in range(self.num_classes):
            _out = torch.cat([out[:,:k,:,:] , out[:,(k+1):,:,:]],1)
            c_out[:,k,:,:] = -torch.logsumexp(_out, 1)

        # Force the model to update the worst side
        l_out = -torch.logsumexp(torch.stack([-c_out,-out],0),0)

        # Where is the model looking?
        attn = torch.sigmoid(torch.logsumexp(l_out, 1, keepdim=True))
        out = attn*l_out

        return out, attn

    def forward(self, x):
        out, attn = self.pixelwise_classification(x)
        out = out.sum((-2,-1))/attn.sum((-2,-1))
        return out


class ResNetImpBGDecoder(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetImpBGDecoder, self).__init__()
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

        # Extract the negative max complement logits, which can be
        # referred to as the implicit estimate
        c_out = torch.zeros_like(out)
        for k in range(self.num_classes):
            _out = torch.cat([out[:,:k,:,:] , out[:,(k+1):,:,:]],1)
            c_out[:,k,:,:] = -torch.logsumexp(_out, 1)

        # Force the model to update the worst side: the explicit or the
        # implicit estimate.
        l_out = -torch.logsumexp(torch.stack([-c_out,-out],0),0)

        # Soft masking of negative pixels, to constrain the image-level
        # classification with an attention mechanism.
        attn = torch.sigmoid(torch.logsumexp(l_out, 1, keepdim=True))
        out = attn*l_out

        return out, attn

    def forward(self, x):
        out, attn = self.pixelwise_classification(x)
        out = out.sum((-2,-1))/attn.sum((-2,-1))
        return out


def ResNet18ImpBG(**kwargs):
    return ResNetImpBG(BasicBlock, [2,2,2,2], **kwargs)

def ResNet18ImpBGDecoder(**kwargs):
    return ResNetImpBGDecoder(BasicBlock, [2,2,2,2], **kwargs)

def ResNet34ImpBG(**kwargs):
    return ResNetImpBG(BasicBlock, [3,4,6,3], **kwargs)

def ResNet50ImpBG(**kwargs):
    return ResNetImpBG(Bottleneck, [3,4,6,3], **kwargs)

def ResNet101ImpBG(**kwargs):
    return ResNetImpBG(Bottleneck, [3,4,23,3], **kwargs)

def ResNet152ImpBG(**kwargs):
    return ResNetImpBG(Bottleneck, [3,8,36,3], **kwargs)


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
