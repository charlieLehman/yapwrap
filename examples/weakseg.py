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
        "class":FlickerClean,
        "params":{
            "root":"/data/datasets/FlickrClean",
            "size":224,
            "batch_sizes":{
            "train":100,
            "test":100,
        }}
    },
    "model":{
        "class":ImpAttn50,
        # "class":TinyImpAttn18,
        "params":{
            "optimizer_config":{
                "class_params":{
                    "lr":1e-1,
                    "weight_decay":5e-4,
                },
                "attention_params":{
                    "lr":1e-3,
                    "weight_decay":5e-4,
                },
                "optimizer":{
                    "class":torch.optim.SGD,
                    "params":{"lr":0.1,
                        "momentum":0.9,
                              "nesterov":True}
                }
            }
        }
    },
    "lr_scheduler":
    {
        "class":torch.optim.lr_scheduler.CosineAnnealingLR,
        "params":{"T_max":90}
    },
    "criterion":{
        "class":FocalLoss,
        "params":{
            "gamma":1,
            "alpha":None,
        }
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
    "visualize_every_n_step":20,
    "max_visualize_batch":16,
    "visualize_every_epoch":True,
    }
exp = ImpBGClassification(config)
exp.train(90)
