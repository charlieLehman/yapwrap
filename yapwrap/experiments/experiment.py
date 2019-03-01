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
import re
from tqdm import tqdm

class Experiment(object):
    def __init__(self, **kwargs):
        r = re.compile('\/run\/([^\/]*)\/experiment_')
        self.name = self.__class__.__name__
        self.experiment_dir = kwargs.get('experiment_dir', None)
        model, optimizer = self._maybe_resume()
        self.logger = yapwrap.utils.Logger(self.experiment_name, self.experiment_dir, **kwargs)
        dataloader = kwargs['dataloader']
        lr = kwargs['lr']
        evaluator = kwargs['evaluator']
        criterion = kwargs['criterion']
        criterion.__str__ = re.sub('[()]','',criterion.__class__.__name__)
        self.saver.save_config(**kwargs)

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

    def _maybe_resume(self):
        if  self.experiment_dir is not None:
            if not r.match(self.experiment_dir):
                print('{} is not an experiment generated by YAPWrap or is malformed.'.format(self.experiment_dir))
                raise

            self.experiment_name = self.experiment_dir.split('run/')[-1].split('/experiment')[0]
            self.saver = yapwrap.utils.Saver(self.experiment_name, self.experiment_dir)
            self.saver.resume()
            kwargs = self.saver.load_config()
            model = kwargs['model']
            optimizer = kwargs['optimizer']
            model.load_state_dict(self.saver.model_state_dict)
            optimizer.load_state_dict(self.saver.optimizer_state_dict)
            self.resumed = True
        else:
            self.experiment_name = '{}_{}'.format(model.name, dataloader.name)
            self.experiment_dir = self._experiment_dir(self.experiment_name)
            self.saver = yapwrap.utils.Saver(self.experiment_name, self.experiment_dir)
            model = kwargs['model']
            optimizer = kwargs['optimizer']
            self.resumed = False
        return model, optimizer

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


