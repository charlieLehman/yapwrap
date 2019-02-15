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

from torch import nn
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def to_np(x):
    return x.detach().cpu().numpy()

class Metric(nn.Module):
    def __init__(self):
        super(Metric, self).__init__()
        self.name = self.__class__.__name__

    def forward(self):
        raise NotImplementedError

    def __str__(self):
        return self.name

    def reset(self):
        pass

class RunningAccuracy(Metric):
    def __init__(self):
        super(RunningAccuracy, self).__init__()
        self.correct = 0
        self.num_examples = 0
        self.acc = 0

    def forward(self, output, target):
        prediction = output.argmax(1)
        self.correct += prediction.eq(target).sum().item()
        self.num_examples += target.size(0)
        self.acc = self.correct/self.num_examples
        return {'RunningAccuracy':self.acc}

    def reset(self):
        self.correct = 0
        self.num_examples = 0

class Accuracy(Metric):
    def __init__(self):
        super(Accuracy, self).__init__()
        self.val = 0

    def forward(self, output, target):
        prediction = output.argmax(1)
        self.val = (prediction.eq(target)).float().mean().item()
        return {'Accuracy':self.val}

class RunningConfusionMatrix(Metric):
    def __init__(self, num_classes):
        super(RunningConfusionMatrix, self).__init__()
        self.matrix = None
        self.labels = range(num_classes)

    def forward(self, output, target):
        prediction = to_np(output.argmax(1))
        target = to_np(target)
        if self.matrix is None:
            self.matrix = confusion_matrix(target, prediction, self.labels)
        else:
            self.matrix += confusion_matrix(target, prediction, self.labels)
        return {'RunningConfusionMatrix':self.matrix}

    def reset(self):
        self.matrix = None


class Evaluator(object):
    def __init__(self, metrics):
        if not isinstance(metrics, dict):
            raise TypeError('{} is not a valid dictionary'.format(type(metrics).__name__))
        for ms in metrics.values():
            for v in ms.values():
                if not isinstance(v, Metric):
                    raise TypeError('{} is not a valid Metric'.format(type(v).__name__))
        self.metrics = metrics
        self.state = {'train':{}, 'validation':{}, 'test':{}}

    def update(self):
        raise NotImplementedError

    def reset(self):
        for v in self.metrics.values():
            for metric in v.values():
                metric.reset()

    def __str__(self):
        return str(self.metrics)

class ImageClassificationEvaluator(Evaluator):
    def __init__(self, num_classes):
        metrics = {
            'train':
            {
                'Accuracy':Accuracy(),
            },
            'validation':
            {
                'RunningAccuracy':RunningAccuracy(),
            },
            'test':
            {
                'TestMetric':RunningAccuracy(),
            },
        }
        super(ImageClassificationEvaluator, self).__init__(metrics)

    def update(self, output, target):
        for metric in self.metrics[self.metric_set].values():
            self.state[self.metric_set].update(metric(output, target))


