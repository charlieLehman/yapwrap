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
from yapwrap.evaluators import Metric

class Evaluator(object):
    def __init__(self, metrics):
        if not isinstance(metrics, dict):
            raise TypeError('{} is not a valid dictionary'.format(type(metrics).__name__))
        for ms in metrics.values():
            for v in ms.values():
                if not isinstance(v, Metric):
                    raise TypeError('{} is not a valid Metric'.format(type(v).__name__))
        self.metrics = metrics
        self.metric_set = None
        self.criterion = None
        self.state = {'train':{}, 'validation':{}, 'test':{}}
        self.step = 0

    def update(self, **kwargs):
        if 'criterion' in kwargs and 'loss' in kwargs:
            self.criterion = kwargs['criterion']
            self.state[self.metric_set].update({self.criterion:kwargs['loss']})
        if 'metrics' in kwargs:
            output, target = kwargs['metrics']
            for metric in self.metrics[self.metric_set].values():
                self.state[self.metric_set].update(metric(output, target))
        if 'values' in kwargs:
            values = kwargs['values']
            for name, value in values.items():
                self.state[self.metric_set].update({name:value})

    def reset(self):
        for v in self.metrics.values():
            for metric in v.values():
                metric.reset()

    def __str__(self):
        return str(self.metrics)

    def tbar_desc(self, epoch):
        desc = "Epoch {} - ".format(epoch)
        if self.criterion is not None:
            desc += "{} - Loss: {:.3f}".format(self.metric_set, self.state[self.metric_set][self.criterion])
        else:
            k,v = self.state[self.metric_set].items()[0]
            desc += "{} - {}: {:.3f}".format(self.metric_set, k,v)
        return desc
