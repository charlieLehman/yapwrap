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
from yapwrap.utils import Metric,  Evaluator

def to_np(x):
    return x.detach().cpu().numpy()

class RunningAccuracy(Metric):
    def __init__(self):
        super(RunningAccuracy, self).__init__()
        self.correct = 0
        self.num_examples = 0

    def forward(self, output, target):
        prediction = output.argmax(1)
        self.correct += prediction.eq(target).sum().item()
        self.num_examples += target.size(0)
        acc = self.correct/self.num_examples
        return {'Accuracy':acc}

    def reset(self):
        self.correct = 0
        self.num_examples = 0

class Accuracy(Metric):
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, output, target):
        prediction = output.argmax(1)
        val = (prediction.eq(target)).float().mean().item()
        return {'Accuracy':val}

class RunningErrorRate(RunningAccuracy):
    def __init__(self):
        super(RunningErrorRate, self).__init__()

    def forward(self, output, target):
        accdict = super(RunningErrorRate, self).forward(output, target)
        acc = accdict.get('Accuracy', 0)
        return {'ErrorRate':1-acc}

class ErrorRate(Accuracy):
    def __init__(self):
        super(ErrorRate, self).__init__()

    def forward(self, output, target):
        accdict = super(ErrorRate, self).forward(output, target)
        acc = accdict.get('Accuracy', 0)
        return {'ErrorRate':1-acc}

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
        return {'ConfusionMatrix':self.matrix}

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
        confidences, predictions = torch.max(F.softmax(output, dim=1), 1)
        accuracies = predictions.eq(target)
        self.num_samples += output.size(0)

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.sum().item()
            self.prop_in_bin += prop_in_bin
            if prop_in_bin > 0:
                self.corr_in_bin += accuracies[in_bin].sum().item()
                self.conf_in_bin += confidences[in_bin].sum().item()
        ece = self.ece()
        return {'ExpectedCalibrationError':ece}

    def ece(self):
        return np.abs(self.corr_in_bin/self.num_samples
                         - self.conf_in_bin/self.num_samples) * (self.prop_in_bin/self.num_samples)


class ImageClassificationEvaluator(Evaluator):
    def __init__(self, num_classes):
        metrics = {
            'train':
            {
                'Accuracy':Accuracy(),
            },
            'validation':
            {
                'Accuracy':RunningAccuracy(),
                'ExpectedCalibrationError':RunningExpectedCalibrationError(),
            },
            'test':
            {
                'TestMetric':RunningAccuracy(),
                'ConfusionMatrix':RunningConfusionMatrix(num_classes),
                'ExpectedCalibrationError':RunningExpectedCalibrationError(),
            },
        }
        super(ImageClassificationEvaluator, self).__init__(metrics)
