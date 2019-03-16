import torch
from torch import nn
from yapwrap.dataloaders import *
from yapwrap.experiments import OutOfDistribution
from yapwrap.utils import OODEvaluator, BestMetricSaver
from yapwrap.utils.lr_scheduler import PolyLR
from yapwrap.models import *
import inspect


config = lambda model, dataloader, ood_dataloaders, num_epochs: {
    "experiment_dir":None,
    "dataloader":{
        "class":dataloader,
        "params":{}
    },
    "ood_dataloaders":{
        "class":OODDataloaders,
        "params":{"dataloader_list":ood_dataloaders}
    },
    "model":{
        "class":model,
        "params":{}
    },
    "lr_scheduler":{
        "class":torch.optim.lr_scheduler.CosineAnnealingLR,
        "params":{"T_max":num_epochs}
    },
    "criterion":{
        "class":nn.CrossEntropyLoss,
        "params":{}
    },
    "evaluator":{
        "class":OODEvaluator,
        "params":{}
    },
    "saver":{
        "class":BestMetricSaver,
        "params":{"metric_set":"validation",
                  "metric_name":"Accuracy"}
    },
    "cuda":True,
    }


# Training Data
dataloaders = [TinyImageNet]
# Out of Distribution Data
_ood_dataloaders = [OOD_TinyImageNet(), OOD_CIFAR10(), OOD_CIFAR100(), OOD_SVHN(), Noise(noise_type='Gaussian'), Noise(noise_type='Rademacher'), Noise(noise_type='Blob')]
_num_epochs = [100]

for i, (dataloader, num_epochs) in enumerate(zip(dataloaders, _num_epochs)):
    ood_dataloaders = [*_ood_dataloaders[:i], *_ood_dataloaders[(i+1):]]

    # Models to Compare
    trn_res = TinyResNet18
    trn_wrn = TinyWideResNet40x2
    trn_attn = TinyAttention18
    trn_segm = TinySegmentation18
    trn_wrnattn = TinyWideAttention40x2
    trn_wrnsegm = TinyWideSegmentation40x2
    models = [
        trn_wrn,
        trn_attn,
        trn_segm,
        trn_wrnattn,
        trn_wrnsegm,
        trn_res,
    ]

    # Run an experiment for each model
    for model in models:
        # Experiment Parameters
        exp = OutOfDistribution(config(model, dataloader, ood_dataloaders, num_epochs))
        exp.train(num_epochs)
