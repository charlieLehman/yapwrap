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

class mIOU(Metric):
    def __init__(self, num_classes):
        super(mIOU, self).__init__()
        self.num_classes = num_classes

    def forward(self, output, target):
        ious = []
        if output.size(1) > 1:
            pred = output.argmax(1).view(-1)
        else:
            pred = (output >= 0.5).view(-1)
        target = target.view(-1)
        for n in range(self.num_classes):
            o_idx = pred == n
            t_idx = target == n
            intersection = (o_idx[t_idx]).long().sum().item()
            union = o_idx.long().sum().item() + t_idx.long().sum().item()
        return {'mIOU':intersection / union}

class RunningmIOU(Metric):
    def __init__(self, num_classes):
        super(RunningmIOU, self).__init__()
        self.intersection = np.zeros(num_classes)
        self.union = np.zeros(num_classes)
        self.num_classes = num_classes

    def forward(self, output, target):
        ious = []
        if output.size(1) > 1:
            pred = output.argmax(1).view(-1)
        else:
            pred = (output >= 0.5).view(-1)
        target = target.view(-1)
        for n in range(self.num_classes):
            o_idx = pred == n
            t_idx = target == n
            self.intersection[n] += (o_idx[t_idx]).long().sum().item()
            self.union[n] += o_idx.long().sum().item() + t_idx.long().sum().item()
        return {'mIOU':(self.intersection / self.union).mean()}

    def reset(self):
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)

class RunningAccuracy(Metric):
    def __init__(self):
        super(RunningAccuracy, self).__init__()
        self.correct = 0
        self.num_examples = 0

    def forward(self, output, target):
        self.num_examples += target.size(0)*target.size(2)*target.size(3)
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

class ImageSegmentationEvaluator(Evaluator):
    def __init__(self, num_classes):
        metrics = {
            'train':
            {
                'mIOU':mIOU(num_classes),
                'Accuracy':Accuracy(),
            },
            'validation':
            {
                'mIOU':RunningmIOU(num_classes),
                'Accuracy':RunningAccuracy(),
            },
            'test':
            {
                'mIOU':RunningmIOU(num_classes),
                'Accuracy':RunningAccuracy(),
            },
        }

        super(ImageSegmentationEvaluator, self).__init__(metrics)
