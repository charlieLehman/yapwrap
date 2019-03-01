import torch
from torch import nn
from yapwrap.dataloaders import *
from yapwrap.experiments import OutOfDistribution
from yapwrap.utils import OODEvaluator
from yapwrap.utils.lr_scheduler import PolyLR
from yapwrap.models import *
import inspect

# Training Data
dataloaders = [CIFAR10(), CIFAR100(), SVHN(), TinyImageNet()]
# Out of Distribution Data
_ood_dataloaders = [OOD_CIFAR10(), OOD_CIFAR100(), OOD_SVHN(), OOD_TinyImageNet(), Noise(noise_type='Gaussian'), Noise(noise_type='Rademacher'), Noise(noise_type='Blob')]

for i, dataloader in enumerate(dataloaders):
    ood_dataloaders = [*_ood_dataloaders[:i], *_ood_dataloaders[(i+1):]]

    # Models to Compare
    trn_a = TinyAttention18(num_classes=dataloader.num_classes)
    trn_ac = ComplementConstraint(TinyAttention18(num_classes=dataloader.num_classes))
    models = [trn_ac, trn_a]

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
