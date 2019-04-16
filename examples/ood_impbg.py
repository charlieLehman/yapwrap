from yapwrap.experiments import ImpBGOOD
from yapwrap.dataloaders import *
from yapwrap.utils import *
from yapwrap.models import *
import torch
import torch.nn as nn

ood_dataloaders = [OOD_CIFAR10(), OOD_TinyImageNet(), OOD_SVHN(), Noise_Gaussian(), Noise_Rademacher(), Noise_Blob()]
config = {
    "experiment_dir":'.',
    "dataloader":{
        "class":CIFAR100,
        "params":{}
    },
    "model":{
        "class":TinyImpBG18,
        "params":{}
    },
    "lr_scheduler":{
        "class":torch.optim.lr_scheduler.CosineAnnealingLR,
        "params":{"T_max":100}
    },
        "ood_dataloaders":{
        "class":OODDataloaders,
        "params":{"dataloader_list":ood_dataloaders}
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
exp = ImpBGOOD(config)
exp.train(100)
