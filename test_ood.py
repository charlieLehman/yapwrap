import torch
from torch import nn
from yapwrap.dataloaders import *
from yapwrap.experiments import OutOfDistribution
from yapwrap.utils import OODEvaluator
from yapwrap.utils.lr_scheduler import PolyLR
from yapwrap.models import *
import inspect

# Training Data
dataloader = CIFAR10()
ood_dataloaders = [OOD_CIFAR100(), OOD_SVHN(), OOD_TinyImageNet(), Noise(noise_type='Gaussian'), Noise(noise_type='Rademacher'), Noise(noise_type='Blob')]

# Models to Compare
trn = TinyResNet18(dataloader.num_classes)
trn_cc = ComplementConstraint(TinyResNet18(dataloader.num_classes))
trn_ccc = ComplementConstraintCombined(TinyResNet18(dataloader.num_classes))
models = [trn, trn_ccc, trn_cc]

# Evaluation Criterion

# Run both experiments
lr = 0.1
num_epochs = 300
for model in models:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=False)
    lr_scheduler = PolyLR(optimizer, num_epochs)
    criterion = nn.CrossEntropyLoss()
    evaluator = OODEvaluator(dataloader.num_classes)
    kwargs = {'model':model, 'lr':lr, 'optimizer':optimizer, 'criterion':criterion, 'lr_scheduler':lr_scheduler, 'dataloader':dataloader, 'ood_dataloaders':ood_dataloaders, 'evaluator':evaluator}
    exp = OutOfDistribution(**kwargs).cuda()
    exp.train(num_epochs)
# exp.test()
