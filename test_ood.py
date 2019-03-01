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

# Out of Distribution Data
ood_dataloaders = [OOD_CIFAR100(), OOD_SVHN(), OOD_TinyImageNet(), Noise(noise_type='Gaussian'), Noise(noise_type='Rademacher'), Noise(noise_type='Blob')]

# Models to Compare
trn = TinyResNet18(dataloader.num_classes)
trn_cc = ComplementConstraint(TinyResNet18(dataloader.num_classes))
trn_ccc = ComplementConstraintCombined(TinyResNet18(dataloader.num_classes))
trn_a = TinyAttention18(num_classes=dataloader.num_classes)
trn_ac = ComplementConstraint(TinyAttention18(num_classes=dataloader.num_classes))
# models = [trn, trn_ccc, trn_cc]
models = [trn_ac, trn_cc]

# Run an experiment for each model
num_epochs = 100
lr = 0.1
for model in models:
    # Optimization Parameters
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    # lr_scheduler = PolyLR(optimizer, num_epochs)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs,1e-6/lr)
    criterion = nn.CrossEntropyLoss()

    # Evaluation Criterion
    evaluator = OODEvaluator(dataloader.num_classes)

    # Experiment Parameters
    kwargs = {'model':model, 'lr':lr, 'optimizer':optimizer, 'criterion':criterion, 'lr_scheduler':lr_scheduler, 'dataloader':dataloader, 'ood_dataloaders':ood_dataloaders, 'evaluator':evaluator}
    exp = OutOfDistribution(**kwargs).cuda()
    exp.train(num_epochs)
