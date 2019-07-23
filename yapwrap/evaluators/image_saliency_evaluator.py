# Copyright Charlie Lehman. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from torch.nn import functional as F
from torch import nn
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, fbeta_score
from yapwrap.evaluators import *

def to_np(x):
    return x.detach().cpu().numpy()


def tp_fp_tn_fn(pred, target):
    target = (target==1)
    tp = pred[target].sum().item()
    fp = (1-target)[pred].sum().item()
    tn = (1-pred)[1-target].sum().item()
    fn = target[1-pred].sum().item()
    return tp, fp, tn, fn

class MAE(Metric):
    def __init__(self):
        super(MAE, self).__init__()

    def forward(self, output, target):
        return {'MAE':torch.abs(output-target).mean().item()}

class RunningMAE(Metric):
    def __init__(self):
        super(RunningMAE, self).__init__()
        self.reset()

    def forward(self, output, target):
        self.count += target.numel()
        self.total += torch.abs(output-target).sum().item()
        return {'MAE':self.total/self.count}

    def reset(self):
        self.count = 0
        self.total = 0

class FBetaScore(Metric):
    def __init__(self, beta2=0.3):
        super(FBetaScore, self).__init__()
        self.beta2 = beta2

    def forward(self, output, target):
        if output.size(1) > 1:
            pred = output.argmax(1).view(-1)
        else:
            pred = (output >= 0.5).view(-1)
        target = target.view(-1)
        tp, fp, tn, fn = tp_fp_tn_fn(pred, target)
        precision = tp/(tp+fp) if tp>0 else 0
        recall = tp/(tp+fn) if tp>0 else 0
        f_beta = (1+self.beta2)*precision*recall/(self.beta2*precision+recall) if tp>0 else 0
        return {'FBeta':f_beta}

    def reset(self):
        self.intersection = []
        self.union = []

class RunningFBetaScore(Metric):
    def __init__(self, beta2=0.3):
        super(RunningFBetaScore, self).__init__()
        self.reset()
        self.beta2 = beta2

    def forward(self, output, target):
        if output.size(1) > 1:
            pred = output.argmax(1).view(-1)
        else:
            pred = (output >= 0.5).view(-1)
        target = target.view(-1)
        tp, fp, tn, fn = tp_fp_tn_fn(pred, target)
        self.tp += tp
        self.fp += fp
        self.tn += tn
        self.fn += fn
        if (self.tp+self.fp) != 0:
            precision = self.tp/(self.tp+self.fp)
        else:
            precision = 0
        if (self.tp+self.fn) != 0:
            recall = self.tp/(self.tp+self.fn)
        else:
            recall = 0
        if recall+precision != 0:
            f_beta = (1+self.beta2)*precision*recall/(self.beta2*precision+recall)
        else:
            f_beta = 0
        return {'FBeta':f_beta}

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

class RunningAccuracy(Metric):
    def __init__(self):
        super(RunningAccuracy, self).__init__()
        self.correct = 0
        self.num_examples = 0

    def forward(self, output, target):
        self.num_examples += target.numel()
        if output.size(1) > 1:
            prediction = output.argmax(1)
        else:
            prediction = (output >= 0.5).int()
        self.correct += prediction.eq(target.int()).sum().item()
        acc = self.correct/self.num_examples
        return {'Accuracy':acc}

    def reset(self):
        self.correct = 0
        self.num_examples = 0

class Accuracy(Metric):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, output, target):
        if output.size(1) > 1:
            prediction = output.argmax(1)
        else:
            prediction = (output >= 0.5).int()
        val = (prediction.eq(target.int())).float().mean().item()
        return {'Accuracy':val}

class ImageSaliencyEvaluator(Evaluator):
    def __init__(self, num_classes):
        metrics = {
            'train':
            {
                'Accuracy':Accuracy(),
                'FBeta':FBetaScore(),
                'MAE':MAE(),
            },
            'validation':
            {
                'Accuracy':RunningAccuracy(),
                'FBeta':RunningFBetaScore(),
                'MAE':RunningMAE(),
            },
            'test':
            {
                'Accuracy':RunningAccuracy(),
                'FBeta':RunningFBetaScore(),
                'MAE':RunningMAE(),
            },
        }

        super(ImageSaliencyEvaluator, self).__init__(metrics)
