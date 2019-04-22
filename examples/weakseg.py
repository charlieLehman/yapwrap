from yapwrap.experiments import *
from yapwrap.dataloaders import *
from yapwrap.utils import *
from yapwrap.models import *
import torch
import torch.nn as nn

config = {
    "experiment_dir":'.',
    "dataloader":{
        "class":ImageFolder,
        "params":{
            "root":"/data/datasets/ImageNet",
            "size":224,
            "batch_sizes":{
            "train":256,
            "test":100,
        }}
    # "dataloader":{
    #     "class":CIFAR10,
    #     "params":{
    #         "batch_sizes":{
    #         "train":128,
    #         "test":100,}
    #         }
    },
    "model":{
        "class":ImpAttn18,
        # "class":TinyResNet18,
        "params":{}
    },
    "lr_scheduler":{
        "class":torch.optim.lr_scheduler.CosineAnnealingLR,
        "params":{"T_max":90}
    },
    "criterion":{
        "class":nn.CrossEntropyLoss,
        "params":{}
    },
    "evaluator":{
        "class":ImageClassificationEvaluator,
        "params":{}
    },
    "saver":{
        "class":BestMetricSaver,
        "params":{"metric_set":"validation",
                  "metric_name":"Accuracy"}
    },
    "cuda":True,
    "vizualize_every_n_step":100,
    }
exp = ImpBGClassification(config)
# exp = ImageClassification(config)
exp.train(90)
