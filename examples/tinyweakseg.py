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
        "class":CIFAR10,
        "params":{
            "size":32,
            "batch_sizes":{
            "train":128,
            "test":100,}
            }
    },
    "model":{
        "class":TinyImpAttn18,
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
    "visualize_every_n_step":50,
    "max_visualize_batch":100,
    "visualize_every_epoch":True,
    }
exp = ImpBGClassification(config)
# exp = ImageClassification(config)
exp.train(90)
