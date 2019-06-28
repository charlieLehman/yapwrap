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
            "validation":20,
            "test":10,}
        }
    },
    "model":{
        "class":PoolNetResNet50,
        "params":{}
    },
    "lr_scheduler":{
        "class":torch.optim.lr_scheduler.CosineAnnealingLR,
        "params":{"T_max":24}
    },
    "criterion":{
        "class":nn.BCELoss,
        "params":{"reduction":"sum"}
    },
    "evaluator":{
        "class":ImageSaliencyEvaluator,
        "params":{}
    },
    "saver":{
        "class":BestMetricSaver,
        "params":{"metric_set":"validation",
                  "metric_name":"FBeta"}
    },
    "cuda":True,
    "visualize_every_n_step":None,
    "max_visualize_batch":9,
    "visualize_every_epoch":True,
    }
exp = ImageSegmentation(config)
exp.train(24)
