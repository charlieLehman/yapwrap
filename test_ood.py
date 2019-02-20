import torch
from torch import nn
from yapwrap.dataloaders import CIFAR10, SVHN, NoisyDataloader
from yapwrap.experiments import OutOfDistribution
from yapwrap.utils import OODEvaluator
from yapwrap.utils.lr_scheduler import PolyLR
from yapwrap.models import TinyResNet18, ComplementConstraint, TinyResNet50
from torchvision.models import resnet18
import inspect

# Training Data
dataloader = CIFAR10()
ood_dataloaders = [SVHN(), NoisyDataloader(CIFAR10(), p=0.2), NoisyDataloader(CIFAR10(), p=0.5)]

# Models to Compare
trn = TinyResNet18(dataloader.num_classes)
trn_cc = ComplementConstraint(TinyResNet18(dataloader.num_classes))
models = [trn, trn_cc]

# Evaluation Criterion

# Run both experiments
lr = 0.1
num_epochs = 100
for model in models:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=False)
    lr_scheduler = PolyLR(optimizer, num_epochs)
    criterion = nn.CrossEntropyLoss()
    evaluator = OODEvaluator(dataloader.num_classes)
    kwargs = {'model':model, 'lr':lr, 'optimizer':optimizer, 'criterion':criterion, 'lr_scheduler':lr_scheduler, 'dataloader':dataloader, 'ood_dataloaders':ood_dataloaders, 'evaluator':evaluator}
    exp = OutOfDistribution(**kwargs).cuda()
    exp.train(num_epochs)
# exp.test()
