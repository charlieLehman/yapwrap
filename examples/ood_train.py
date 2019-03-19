import torch
from torch import nn
from yapwrap.dataloaders import *
from yapwrap.experiments import OutOfDistribution
from yapwrap.utils import OODEvaluator, BestMetricSaver
from yapwrap.utils.lr_scheduler import PolyLR
from yapwrap.models import *
import inspect


config = lambda model, dataloader, ood_dataloaders, num_epochs: {
    "experiment_dir":'/media/advait/DATA/Advait/Handouts_and_assignments/Spring_2019_coursework/yapwrap/run/TinyAttention18_CIFAR10',
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
dataloaders = [CIFAR10]
# Out of Distribution Data
_ood_dataloaders = [OOD_CIFAR10(), OOD_TinyImageNet()]
_num_epochs = [1]

for i, (dataloader, num_epochs) in enumerate(zip(dataloaders, _num_epochs)):
    ood_dataloaders = [*_ood_dataloaders[:i], *_ood_dataloaders[(i+1):]]

    # Models to Compare
    trn_attn = TinyAttention18
    models = [
        trn_attn,
    ]

    # Run an experiment for each model
    for model in models:
        # Experiment Parameters
        exp = OutOfDistribution(config(model, dataloader, ood_dataloaders, num_epochs))
        exp.train(num_epochs)
        exp._ood_test()
