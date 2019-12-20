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

import yapwrap
import torch
from torch.optim import *
from torch import nn
import os
import glob
import re
import importlib
import json
from tqdm import tqdm

class Experiment(object):
    def __init__(self, config, cuda=False):
        self.make_logs = True
        self.resumed = False
        self.on_cuda = False
        self.config = config
        self.name = self.__class__.__name__
        self.experiment_dir = None
        self._maybe_resume()
        self.visualize_every_n_step = config.get('visualize_every_n_step', None)
        self.max_visualize_batch = config.get('max_visualize_batch', None)
        self.visualize_every_epoch = config.get('visualize_every_epoch', True)
        self.logger = yapwrap.loggers.Logger(self.experiment_name, self.experiment_dir)

        if not isinstance(self.model, nn.Module):
            raise TypeError('{} is not a valid type nn.Module'.format(type(self.model).__name__))
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not a valid optimizer'.format(type(self.optimizer).__name__))
        # if not (isinstance(self.criterion, nn.modules.loss._Loss):
        #     raise TypeError('{} is not a valid criterion'.format(type(self.criterion).__name__))
        if not isinstance(self.evaluator, yapwrap.evaluators.Evaluator):
            raise TypeError('{} is not a valid yapwrap.evaluators.Evaluator'.format(type(self.evaluator).__name__))
        if not isinstance(self.dataloader, yapwrap.dataloaders.Dataloader):
            raise TypeError('{} is not a valid type yapwrap.dataloaders.Dataloader'.format(type(self.dataloader).__name__))

    def _maybe_resume(self):
        ## Dataloader
        self.dataloader = self.config['dataloader']['class'](**self.config['dataloader']['params'])

        ## Model
        model_config = self.config['model']['params']
        model_config.update({'num_classes':self.dataloader.num_classes})
        self.model = self.config['model']['class'](**model_config)
        self.experiment_name = '{}_{}'.format(self.model.name, self.dataloader.name)
        run_id = self.config.get('experiment_number', None)
        self.experiment_dir = get_experiment_dir(self.experiment_name, run_id)
        ## Config and Saver
        if not os.path.exists(self.experiment_dir):
            assert run_id is None, 'experiment_number defined and {} does not exist.'.format(self.experiment_dir)
            print('Created {}'.format(self.experiment_dir))
            os.makedirs(self.experiment_dir)
            saver_ = self.config['saver']['class']
            self.saver = saver_( experiment_name=self.experiment_name, experiment_dir=self.experiment_dir, **self.config['saver']['params'])
            self.saver.save_config(self.config)

        if run_id is not None:
            state = torch.load(os.path.join(self.experiment_dir, 'checkpoint.pth.tar'))
            print('Loading Model from {} at epoch {}'.format(self.experiment_dir, state['epoch']))
            self.model.load_state_dict(state['model_state_dict'])
        try:
            self.config.update({'flops':yapwrap.utils.get_model_complexity_info(self.model,self.dataloader.size, False, True)[0]})

        except:
            pass

        ## Criterion
        self.criterion = self.config['criterion']['class'](**self.config['criterion']['params'])

        ## CUDA
        if self.config.get('cuda', False):
            self.model = nn.DataParallel(self.model)
            self.model.cuda()
            self.model.name = self.model.module.name
            self.on_cuda = True

        ## Optimizer
        self._init_optimizer()

        ## LR Scheduler
        self._init_lr_scheduler()

        ## Evaluator
        self.evaluator = self.config['evaluator']['class'](num_classes=self.dataloader.num_classes, **self.config['evaluator']['params'])


    def _init_lr_scheduler(self):
        if self.config.get('lr_scheduler', None) is None:
            self.lr_scheduler = None
            return
        self.lr_scheduler = self.config['lr_scheduler']['class'](optimizer=self.optimizer, **self.config['lr_scheduler']['params'])

    def _init_optimizer(self):
        _model = self.model.module if self.on_cuda else self.model

        if not hasattr(_model, 'optimizer_config'):
            raise NotImplementedError('{} does not have a optimizer_config.'.format(_model.name))

        _opt_conf = _model.optimizer_config
        _opt = _opt_conf['class']
        model_params = getattr(_model, 'optimizer_parameters', self.model.parameters)()
        self.optimizer = _opt(model_params, **_opt_conf['params'])


    def save(self):
        self.saver.save()

    def _step(self):
        raise NotImplementedError

    def _epoch(self):
        raise NotImplementedError

    def __str__(self):
        return str(self.model)

    def _get_model_state(self):
        if self.on_cuda:
            return self.model.module.state_dict()
        return self.model.state_dict()

    def visualize(self):
        raise NotImplementedError


class NotExperimentError(Exception):
    def __init__(self, message):
        self.message = message

def get_experiment_dir(experiment_name, run_id=None):
    if experiment_name is None:
        return None
    directory = os.path.join('run', experiment_name)
    runs = sorted(glob.glob(os.path.join(directory, 'experiment_*')))
    if run_id is None:
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
    exp_dir = os.path.join(directory, 'experiment_{:04d}'.format(run_id))
    return exp_dir
