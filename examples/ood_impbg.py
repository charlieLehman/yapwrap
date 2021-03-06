from yapwrap.experiments import ImpBGOOD
from yapwrap.dataloaders import *
from yapwrap.utils import *
from yapwrap.models import *
import torch
import torch.nn as nn

ood_dataloaders = [OOD_CIFAR100(), OOD_TinyImageNet(), OOD_SVHN(), Noise_Gaussian(), Noise_Rademacher(), Noise_Blob()]
config = {
    "experiment_dir":'.',
    "dataloader":{
        "class":CIFAR10,
        "params":{"batch_sizes":{
            "train":128,
            "test":100,
        }}
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
