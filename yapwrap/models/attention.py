import torch
import torch.nn as nn

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

class Attention(nn.Module):
    def __init__(self, num_classes, downsamples=4):
        super(Attention, self).__init__()
        self.name = 'Attention'
        self.first = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=3, padding=1, stride=1, bias=False),
            nn.ReLU(),
        )
        self.upsample = lambda x, s: nn.functional.interpolate(x, s, mode='bilinear', align_corners=True)
        self.downsample = nn.MaxPool2d(2,2)
        self.fuse = nn.Conv2d(downsamples,1,1, bias=False)
        self.downsamples = downsamples
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        s = (x.size(2), x.size(3))
        outs = []
        _x = self.first(x)
        for size in range(self.downsamples):
            _x = self.downsample(_x)
            out = self.conv(_x)
            outs.append(self.upsample(out,s))
        outs.append(self.fuse(torch.relu(torch.cat(outs,1))))
        output = torch.sigmoid(torch.stack(outs))
        print(len(outs))
        print(output.shape)
        return output

    def optimizer_parameters(self):
        return self.parameters()

    @property
    def optimizer_config(self):
        return {"class":torch.optim.SGD,
                "params":{"lr":0.1,
                        "momentum":0.9,
                              "nesterov":True}
               }

    def visualize(self, x, target):
        out = self.forward(x)
        x -= x.min()
        x /= x.max()
        viz_dict = {'Input':x,
                    'Attention':out,
                    'Ground Truth':target,
        }

        return viz_dict
