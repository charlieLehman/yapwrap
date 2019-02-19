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
from torch import nn
import os
import glob
from tqdm import tqdm

class Experiment(object):
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__

        model = kwargs['model']
        dataloader = kwargs['dataloader']
        lr = kwargs['lr']
        evaluator = kwargs['evaluator']
        optimizer = kwargs['optimizer']
        criterion = kwargs['criterion']
        criterion.__str__ = criterion.__class__.__name__.split('(')[0]
        self.experiment_name = '{}_{}'.format(model.name, dataloader.name)
        self.experiment_dir = self._experiment_dir(self.experiment_name)
        self.saver = yapwrap.utils.Saver(self.experiment_name, self.experiment_dir)
        self.saver.experiment_config(**kwargs)
        self.logger = yapwrap.utils.Logger(self.experiment_name, self.experiment_dir, **kwargs)

        if not isinstance(model, nn.Module):
            raise TypeError('{} is not a valid type nn.Module'.format(type(model).__name__))
        self.model = model
        if not isinstance(lr, (int, float)):
            raise TypeError('{} is not a valid learning rate'.format(type(lr).__name__))
        self.lr = lr
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not a valid optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        if not isinstance(criterion, nn.modules.loss._Loss):
            raise TypeError('{} is not a valid criterion'.format(type(criterion).__name__))
        self.criterion = criterion
        if not isinstance(evaluator, yapwrap.utils.Evaluator):
            raise TypeError('{} is not a valid pytorchlab.Evaluator'.format(type(evaluator).__name__))
        self.evaluator = evaluator
        if not isinstance(dataloader, yapwrap.dataloaders.Dataloader):
            raise TypeError('{} is not a valid type pytorchlab.Dataloader'.format(type(dataloader).__name__))
        self.dataloader = dataloader
        self.on_cuda = False

    @staticmethod
    def _experiment_dir(experiment_name):
        directory = os.path.join('run', experiment_name)
        runs = sorted(glob.glob(os.path.join(directory, 'experiment_*')))
        run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0
        exp_dir = os.path.join(directory, 'experiment_{:04d}'.format(run_id))
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
        return exp_dir

    def save(self):
        self.saver.save()

    def _step(self):
        raise NotImplementedError

    def _epoch(self):
        raise NotImplementedError

    def cuda(self):
        self.on_cuda = True
        self.model = nn.DataParallel(self.model).cuda()
        return self

    def __str__(self):
        return str(self.model)

    def _get_model_state(self):
        if self.on_cuda:
            return self.model.module.state_dict()
        return self.model.state_dict()


