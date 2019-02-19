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

import torch
from torch import nn
import numpy as np
import json
import os

class Saver(object):
    def __init__(self, experiment_name, experiment_dir):
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir

        self.loss = None
        self.step = 0
        self.epoch = 0
        self.model_state_dict = None
        self.optimizer_state_dict = None

    def save(self, *args):
        print('======Saving Checkpoint======')
        filename = os.path.join(self.experiment_dir, 'checkpoint.pth.tar')
        state = {'loss':self.loss,
                 'step':self.step,
                 'epoch':self.epoch,
                 'model_state_dict':self.model_state_dict,
                 'optimizer_state_dict':self.optimizer_state_dict}
        torch.save(state, filename)

    def experiment_config(self, **kwargs):
        config = {}
        for k, v in kwargs.items():
            config.update({k:str(v)})
        config.update({'_Experiment_Name_':self.experiment_name})
        filename = os.path.join(self.experiment_dir, 'experiment_config.json')
        with open(filename, 'w') as f:
            json.dump(config, f, sort_keys=True, indent=4)

class BestMetricSaver(Saver):
    def __init__(self, metric_set, metric_name, experiment_name, experiment_dir, criterion=np.greater_equal):
        super(BestMetricSaver, self).__init__(experiment_name, experiment_dir)
        self.metric_set = metric_set
        self.metric_name = metric_name
        self.metric = None
        self.best_metric = None
        self.criterion = criterion

    def save(self, **kwargs):
        self.metric = kwargs['metric_evaluator'].state[self.metric_set][self.metric_name]
        if self.best_metric is None:
            self.best_metric = self.metric
        if self.criterion(self.metric, self.best_metric):
            filename = os.path.join(self.experiment_dir, 'best_{}_checkpoint.pth.tar'.format(self.metric_name))
            state = {'loss':self.loss,
                    'step':self.step,
                    'epoch':self.epoch,
                    'model_state_dict':self.model_state_dict,
                    'optimizer_state_dict':self.optimizer_state_dict}
            state.update({self.metric_name:self.metric})
            torch.save(state, filename)
            print('======Saved Best {} Checkpoint======'.format(self.metric_name))
            self.best_metric = self.metric

class BestLossSaver(Saver):
    def __init__(self, experiment_name, criterion=np.less_equal, **kwargs):
        super(BestLossSaver, self).__init__(experiment_name, **kwargs)
        self.criterion = criterion
        self.best_loss = None

    def save(self, **kwargs):
        if self.best_loss is None:
            self.best_loss = loss
        if self.criterion(self.loss, self.best_loss):
            print('======Saving Best Loss Checkpoint======')
            filename = os.path.join(self.experiment_dir, 'best_loss_checkpoint.pth.tar')
            state = {'loss':self.loss,
                    'step':self.step,
                    'epoch':self.epoch,
                    'model_state_dict':self.model_state_dict,
                    'optimizer_state_dict':self.optimizer_state_dict}
            state.update(kwargs['metric'])
            torch.save(state, filename)
            self.best_loss = loss



