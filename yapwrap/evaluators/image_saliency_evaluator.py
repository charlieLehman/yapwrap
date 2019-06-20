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
from sklearn.metrics import confusion_matrix
from yapwrap.evaluators import Metric,  Evaluator

def to_np(x):
    return x.detach().cpu().numpy()

class mIOU(Metric):
    def __init__(self, num_classes):
        super(mIOU, self).__init__()
        self.intersection = np.zeros(num_classes)
        self.union = np.zeros(num_classes)
        self.num_classes = num_classes

    def forward(self, output, target):
        ious = []
        if output.size(1) > 1:
            pred = output.argmax(1)
        else:
            output = (output > 0.5).view(-1)
        target = target.view(-1)
        for n in range(self.num_classes):
            o_idx = output == n
            t_idx = target == n
            self.intersection[n] += (o_idx[t_idx]).long().sum().item()
            self.union[n] += o_idx.long().sum().item() + t_idx.long().sum().item()
        return {'mIOU':(self.intersection / self.union).mean()}

    def reset(self):
        self.intersection = np.zeros(self.num_classes)
        self.union = np.zeros(self.num_classes)

class ImageSegmentationEvaluator(Evaluator):
    def __init__(self, num_classes):
        metrics = {
            'train':
            {
                'mIOU':mIOU(num_classes),
            },
            'validation':
            {
                'mIOU':mIOU(num_classes),
            },
            'test':
            {
                'mIOU':mIOU(num_classes),
            },
        }

        super(ImageSegmentationEvaluator, self).__init__(metrics)
