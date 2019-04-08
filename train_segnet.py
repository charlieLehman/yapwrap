import torch
from torch import nn
from yapwrap.dataloaders import *
from yapwrap.experiments import ImageSegmentation
from yapwrap.utils import BestMetricSaver, ImageSegmentationEvaluator
from yapwrap.utils.lr_scheduler import PolyLR
from yapwrap.models import *
import inspect

config = lambda model, dataloader, num_epochs: {
    "experiment_dir":'./yapwrap/run/',
    "dataloader":{
        "class":dataloader,
        "params":{'root': './yapwrap/data/VOCdevkit/VOC2012/'}
    },
    "model":{
        "class":model,
        "params":{'args':{'in_channels':3, 'is_unpooling': False}}
    },
    "lr_scheduler":{
        "class":PolyLR,
        "params":{"T_max":num_epochs}
    },
    "optimizer":{
        "class":torch.optim.SGD,
        "params":{'lr':0.01}
    },
    "criterion":{
        "class":nn.CrossEntropyLoss,
        "params":{'reduction':'elementwise_mean', 'ignore_index':255}
    },
    "evaluator":{
        "class":ImageSegmentationEvaluator,
        "params":{}
    },
    "saver":{
        "class":BestMetricSaver,
        "params":{"metric_set":"validation",
                  "metric_name":"Accuracy"}
    },
    "cuda":True,
    }

dataloader = PascalVOC
num_epochs = 1
model = SegNet
exp = ImageSegmentation(config(model, dataloader, num_epochs))
exp.train(num_epochs)
