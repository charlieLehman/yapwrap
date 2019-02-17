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

class RunningExpectedCalibrationError(Metric):
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(RunningExpectedCalibrationError, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.corr_in_bin = 0
        self.conf_in_bin = 0
        self.prop_in_bin = 0
        self.num_samples = 0

    def reset(self):
        self.corr_in_bin = 0
        self.conf_in_bin = 0
        self.prop_in_bin = 0
        self.num_samples = 0

    def forward(self, output, target):
        # labels = labels.long()
        confidences, predictions = torch.max(F.softmax(output, dim=1), 1)
        accuracies = predictions.eq(target)
        self.num_samples += output.size(0)

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.sum().item()
            self.prop_in_bin += prop_in_bin
            if prop_in_bin > 0:
                self.corr_in_bin += accuracies[in_bin].sum().item()
                self.conf_in_bin += confidences[in_bin].sum().item()
        ece = self.ece()
        return {'RunningExpectedCalibrationError':ece}

    def ece(self):
        return np.abs(self.corr_in_bin/self.num_samples
                         - self.conf_in_bin/self.num_samples) * (self.prop_in_bin/self.num_samples)

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

    def reset(self):
        for v in self.metrics.values():
            for metric in v.values():
                metric.reset()

    def __str__(self):
        return str(self.metrics)

    def tbar_desc(self):
        return str(self)

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
                'RunningExpectedCalibrationError':RunningExpectedCalibrationError(),
            },
            'test':
            {
                'TestMetric':RunningAccuracy(),
            },
        }
        super(ImageClassificationEvaluator, self).__init__(metrics)

    def tbar_desc(self, epoch):
        desc = "Epoch {} - ".format(epoch)
        if self.criterion is not None:
            desc += "{} - Loss: {:.3f}".format(self.metric_set, self.state[self.metric_set][self.criterion])
        else:
            k,v = self.state[self.metric_set].items()[0]
            desc += "{} - {}: {:.3f}".format(self.metric_set, k,v)
        return desc



