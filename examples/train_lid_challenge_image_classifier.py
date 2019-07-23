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
            "size":32,
            "batch_sizes":{
            "train":100,
            "val":100,
            "test":100,}
        }
    },
    "model":{
        "class":TinyImpAttn18,
        "params":{
            "pretrained_attn_path":'run/TinyPoolNetResNet18_MSRAB/experiment_0001/checkpoint.pth.tar',
            "pretrained":False,
            "optimizer_config":{
                "class_params":{
                    "lr":5e-5,
                    "weight_decay":5e-4,
                },
                "10x_params":{
                    "lr":5e-4,
                    "weight_decay":5e-4,
                },
                "optimizer":{
                    "class":torch.optim.Adam,
                    "params":{"lr":5e-5,
                    "weight_decay":5e-4,
                    }
                }
            }
        }
    },
    "lr_scheduler":
    {
        "class":torch.optim.lr_scheduler.CosineAnnealingLR,
        "params":{"T_max":24}
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
        "params":{"metric_set":"train",
                  "metric_name":"Accuracy"}
    },
    "cuda":True,
    "visualize_every_n_step":100,
    "max_visualize_batch":16,
    "visualize_every_epoch":True,
    }

exp = LIDChallengeTask1(config)
exp.train(24)

# model = Runner('PoolNetResNet50_MSRAB', 0)
