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
from yapwrap.models import *
from yapwrap.modules import *
from yapwrap.utils import *
from yapwrap.dataloaders import *
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
    def __init__(self, config=None, experiment_name=None, experiment_number=None, cuda=False):
        self.make_logs = True
        self.resumed = False
        self.on_cuda = False
        self.config = config
        self.visualize_every_n_step = config.get('visualize_every_n_step', None)
        self.name = self.__class__.__name__
        self.experiment_dir = get_experiment_dir(experiment_name, experiment_number)
        self._maybe_resume()
        self.distributed_config = config.get('distributed_config', None)
        self.logger = yapwrap.utils.Logger(self.experiment_name, self.experiment_dir)

        if not isinstance(self.model, nn.Module):
            raise TypeError('{} is not a valid type nn.Module'.format(type(self.model).__name__))
        if not isinstance(self.optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not a valid optimizer'.format(type(self.optimizer).__name__))
        # if not (isinstance(self.criterion, nn.modules.loss._Loss):
        #     raise TypeError('{} is not a valid criterion'.format(type(self.criterion).__name__))
        if not isinstance(self.evaluator, yapwrap.utils.Evaluator):
            raise TypeError('{} is not a valid yapwrap.utils.Evaluator'.format(type(self.evaluator).__name__))
        if not isinstance(self.dataloader, yapwrap.dataloaders.Dataloader):
            raise TypeError('{} is not a valid type yapwrap.utils.Dataloader'.format(type(self.dataloader).__name__))

    def _maybe_resume(self):
        r = re.compile('(.*)run\/([^\/]*)\/experiment_(.*)')
        if  self.experiment_dir is not None:
            if not r.match(self.experiment_dir):
                message = '{} is not an experiment generated by YAPWrap or is malformed.'.format(self.experiment_dir)
                raise NotExperimentError(message)

            self.experiment_name = self.experiment_dir.split('run/')[-1].split('/experiment')[0]
            saver_ = yapwrap.utils.Saver(self.experiment_name, self.experiment_dir)
            self.config = saver_.load_config()

            ## Saver
            saver_ = getattr(yapwrap.utils, self.config['saver']['class'])
            self.saver = saver_(experiment_name=self.experiment_name, experiment_dir=self.experiment_dir, **self.config['saver']['params'])
            state = self.saver.resume()

            ## Model
            model_ = getattr(yapwrap.models, self.config['model']['class'])
            self.config['model']['class'] = model_
            self.model = model_(**self.config['model']['params'])
            self.model.load_state_dict(state['model_state_dict'])

            ## Criterion
            criterion_ = getattr(torch.nn, self.config['criterion']['class'])
            self.config['criterion']['class'] = criterion_
            self.criterion = criterion_(**self.config['criterion']['params'])

            ## CUDA
            if self.config.get('cuda', False):
                self.model = nn.DataParallel(self.model)
                self.model.cuda()
                self.model.name = self.model.module.name
                self.on_cuda = True

            ## Dataloader
            dataloader_ = getattr(yapwrap.dataloaders, self.config['dataloader']['class'])
            self.config['dataloader']['class'] = dataloader_
            self.dataloader = dataloader_(**self.config['dataloader']['params'])

            ## Optimizer
            if self.config.get('optimizer', None) is not None:
                optimizer_ = getattr(torch.optim, self.config['optimizer']['class'])
                self.config['optimizer']['class'] = optimizer_
                # self.optimizer = optimizer_(params=self.model.parameters(),**self.config['optimizer']['params'])
            self._init_optimizer()

            self.optimizer.load_state_dict(state_dict=state['optimizer_state_dict'])

            if self.config.get('lr_scheduler', None) is not None:
                lr_scheduler_ = getattr(yapwrap.utils, self.config['lr_scheduler']['class'])
                    # lr_scheduler_ = getattr(torch.optim.lr_scheduler, self.config['lr_scheduler']['class'])
                self.config['lr_scheduler']['class'] = lr_scheduler_
            self._init_lr_scheduler()


            ## Evaluator
            evaluator_ = getattr(yapwrap.utils, self.config['evaluator']['class'])
            self.config['evaluator']['class'] = evaluator_
            self.evaluator = evaluator_(num_classes=self.dataloader.num_classes, **self.config['evaluator']['params'])

            self.resumed = True
        else:
            ## Dataloader
            self.dataloader = self.config['dataloader']['class'](**self.config['dataloader']['params'])

            ## Model
            model_config = self.config['model']['params']
            model_config.update({'num_classes':self.dataloader.num_classes})
            self.model = self.config['model']['class'](**model_config)
            self.config.update({'flops':get_model_complexity_info(self.model,self.dataloader.size, False, True)[0]})

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

            ## Config and Saver
            self.experiment_name = '{}_{}'.format(self.model.name, self.dataloader.name)
            self.experiment_dir = get_experiment_dir(self.experiment_name)
            print('Created {}'.format(self.experiment_dir))
            if not os.path.exists(self.experiment_dir):
                os.makedirs(self.experiment_dir)
            saver_ = self.config['saver']['class']
            self.saver = saver_( experiment_name=self.experiment_name, experiment_dir=self.experiment_dir, **self.config['saver']['params'])

            self.saver.save_config(self.config)

    def _init_lr_scheduler(self):
        if self.config.get('lr_scheduler', None) is None:
            self.lr_scheduler = None
            return
        self.lr_scheduler = self.config['lr_scheduler']['class'](optimizer=self.optimizer, **self.config['lr_scheduler']['params'])

    def _init_optimizer(self):
        _model = self.model.module if self.on_cuda else self.model

        if self.config.get('optimizer', None) is None:
            if not hasattr(_model, 'default_optimizer_config'):
                raise NotImplementedError('{} does not have a default_optimizer, either define one in the model or provide an optimizer configuration in the experiment config.'.format(_model.name))
            self.config.update(_model.default_optimizer_config)

        model_params = getattr(_model, 'default_optimizer_parameters', self.model.parameters)()
        self.optimizer = self.config['optimizer']['class'](model_params, **self.config['optimizer']['params'])


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
