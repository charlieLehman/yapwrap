from yapwrap.experiments import *
from yapwrap.dataloaders import *
from yapwrap.models import *
from yapwrap.modules import *
from yapwrap.evaluators import *
from yapwrap.loggers import *
import torch
import torch.nn as nn

config = {
    "experiment_dir":'.',
    "dataloader":{
        "class":MSRAB,
        "params":{
            "root":'/data/datasets/MSRA-B/',
            "size":224,
            "batch_sizes":{
            "train":20,
            "validation":10,
            "test":10,}
        }
    },
    "model":{
        "class":Attention,
        "params":{}
    },
    "lr_scheduler":{
        "class":torch.optim.lr_scheduler.CosineAnnealingLR,
        "params":{"T_max":100}
    },
    "criterion":{
        "class":nn.BCELoss,
        "params":{}
    },
    "evaluator":{
        "class":ImageSegmentationEvaluator,
        "params":{}
    },
    "saver":{
        "class":BestMetricSaver,
        "params":{"metric_set":"validation",
                  "metric_name":"mIOU"}
    },
    "cuda":True,
    "visualize_every_n_step":None,
    "max_visualize_batch":9,
    "visualize_every_epoch":True,
    }
exp = ImageSegmentation(config)
exp.train(100)
