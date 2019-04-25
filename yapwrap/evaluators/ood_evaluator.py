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
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from yapwrap.evaluators import (Metric,
                           Evaluator,
                           Accuracy,
                           RunningAccuracy,
                           RunningExpectedCalibrationError,
                           RunningConfusionMatrix,
                           ImageClassificationEvaluator)
def to_np(x):
    return x.detach().cpu().numpy()


class OODEvaluator(ImageClassificationEvaluator):
    def __init__(self, num_classes):
        super(OODEvaluator, self).__init__(num_classes)
        self.id_state = (None, None)
        self.ood_state = (None, None)
        self.state.update({'OOD':{'FPR90':{}, 'FPR95':{}, 'AUROC':{}, 'AUPR':{}}})

    def update(self, **kwargs):
        super(OODEvaluator, self).update(**kwargs)
        if kwargs.get('metrics', None) is not None:
            output, target = kwargs['metrics']
            confidence = torch.softmax(output,1).max(1)[0]
            if (self.metric_set == 'validation' or self.metric_set == 'test') and 'metrics' in kwargs:
                target = torch.zeros_like(target)
                if any(x is None for x in self.id_state):
                    self.id_state = (to_np(confidence), to_np(target))
                else:
                    _confidence, _target = self.id_state
                    self.id_state = (np.concatenate((_confidence,to_np(confidence))),
                                    np.concatenate((_target,to_np(target))))

    def ood_update(self, output, target):
        target = torch.ones_like(target)
        confidence = torch.softmax(output,1).max(1)[0]
        if any( x is None for x in self.ood_state):
            self.ood_state = (to_np(confidence), to_np(target))
        else:
            _confidence, _target = self.ood_state
            self.ood_state = (np.concatenate((_confidence, to_np(confidence))),
                              np.concatenate((_target, to_np(target))))

    def ood_run(self, dataloader_name):
        id_confidence, id_target = self.id_state
        ood_confidence, ood_target = self.ood_state
        confidence = np.concatenate((id_confidence, ood_confidence))
        ood_conf = 1 - confidence
        target = np.concatenate((id_target, ood_target))
        name = '{}/FPR95'.format(dataloader_name)
        self.state['OOD']['FPR95'].update({name:fpr_and_fdr_at_recall(target, ood_conf, recall_level=0.95)})
        name = '{}/FPR90'.format(dataloader_name)
        self.state['OOD']['FPR90'].update({name:fpr_and_fdr_at_recall(target, ood_conf, recall_level=0.90)})
        name = '{}/AUROC'.format(dataloader_name)
        self.state['OOD']['AUROC'].update({name:roc_auc_score(target, ood_conf)})
        name = '{}/AUPR'.format(dataloader_name)
        self.state['OOD']['AUPR'].update({name:average_precision_score(target, ood_conf)})
        self.ood_state = (None, None)

    def reset(self):
        super(OODEvaluator, self).reset()
        self.id_state = (None, None)
        self.ood_state = (None, None)


def fpr_and_fdr_at_recall(y_true, y_score, recall_level, pos_label=None):
    '''
    Original source from https://github.com/hendrycks/outlier-exposure
    '''
    
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    '''
    Original source from https://github.com/hendrycks/outlier-exposure
    '''
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out
