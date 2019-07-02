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
        "class":LIDTrack1,
        "params":{
            "root":'/data/datasets/LID',
            "size":224,
            "batch_sizes":{
            "train":16,
            "val":10,
            "test":10,}
        }
    },
    "model":{
        "class":ImpAttn50,
        "params":{
            "pretrained_attn_path":'run/PoolNetResNet50_MSRAB/experiment_0000/checkpoint.pth.tar',
            "optimizer_config":{
                "class_params":{
                    "lr":7e-3,
                    "weight_decay":5e-4,
                },
                "10x_params":{
                    "lr":7e-2,
                    "weight_decay":5e-4,
                },
                "optimizer":{
                    "class":torch.optim.SGD,
                    "params":{"lr":1e-2,
                        "momentum":0.9,
                              "nesterov":False}
                }
            }
        }
    },
    "lr_scheduler":
    {
        "class":torch.optim.lr_scheduler.CosineAnnealingLR,
        "params":{"T_max":20}
    },
    "criterion":{
        "class":nn.CrossEntropyLoss,
        "params":{
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
    "visualize_every_n_step":200,
    "max_visualize_batch":16,
    "visualize_every_epoch":True,
    }

exp = LIDChallengeTask1(config)
exp.train(20)
