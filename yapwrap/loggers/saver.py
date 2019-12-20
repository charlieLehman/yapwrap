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
import re
import copy
import inspect
from yapwrap.utils import recursive_naming

class Saver(object):
    def __init__(self, experiment_name, experiment_dir):
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir

        self.loss = None
        self.step = 0
        self.epoch = 0
        self.model_state_dict = None
        self.optimizer_state_dict = None
        self.exp_path = os.path.join(self.experiment_dir, 'checkpoint.pth.tar')
        self.config_path = os.path.join(self.experiment_dir, 'experiment_config.json')

    def save(self, *args):
        print('======Saving Checkpoint======')
        state = {'loss':self.loss,
                 'step':self.step,
                 'epoch':self.epoch,
                 'model_state_dict':self.model_state_dict,
                 'optimizer_state_dict':self.optimizer_state_dict}
        torch.save(state, self.exp_path)

    def save_config(self, config):
        _config = copy.deepcopy(config)
        _config = recursive_naming(_config)
        with open(self.config_path, 'w') as f:
            json.dump(_config, f, sort_keys=True, indent=4, default=obj_dict)

    def load_config(self):
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        print('Loading Config from {}'.format(self.config_path))
        return config

    def resume(self):
        state = torch.load(self.exp_path)
        self.loss = state['loss']
        self.step = state['step']
        self.epoch = state['epoch']
        self.model_state_dict = state['model_state_dict']
        self.optimizer_state_dict = state['optimizer_state_dict']
        print('Resuming Experiment state from {}'.format(self.exp_path))
        return state

def obj_dict(obj):
    return obj.__dict__

class BestMetricSaver(Saver):
    def __init__(self, metric_set, metric_name, experiment_name, experiment_dir, criterion=np.greater_equal, resume=True):
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
            state = {'loss':self.loss,
                    'step':self.step,
                    'epoch':self.epoch,
                    'model_state_dict':self.model_state_dict,
                     'optimizer_state_dict':self.optimizer_state_dict,
                     self.metric_name:self.metric}
            torch.save(state, self.exp_path)
            print('======Saved Best {} Checkpoint======'.format(self.metric_name))
            self.best_metric = self.metric

    def resume(self):
        state = super(BestMetricSaver, self).resume()
        self.metric = state[self.metric_name]
        self.best_metric = self.metric
        return state

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



