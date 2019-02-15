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

import pytorchlab as pl
from torchvision.models import ResNet
import torch
from torch import nn
from pytorchlab.models import TinyResNet18 as _TinyResNet18
from collections import OrderedDict

class Experiment(object):
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__

        model = kwargs['model']
        dataloader = kwargs['dataloader']
        lr = kwargs['lr']
        evaluator = kwargs['evaluator']
        optimizer = kwargs['optimizer']
        lr_scheduler = kwargs['lr_scheduler']
        criterion = kwargs['criterion']
        criterion.__str__ = criterion.__class__.__name__
        self.experiment_name = '{}_{}'.format(model.name, dataloader.name)

        self.saver = pl.utils.Saver(self.experiment_name, **kwargs)
        self.logger = pl.utils.Logger(self.experiment_name, **kwargs)

        if not isinstance(model, nn.Module):
            raise TypeError('{} is not a valid type nn.Module'.format(type(model).__name__))
        self.model = model
        if not isinstance(lr, (int, float)):
            raise TypeError('{} is not a valid learning rate'.format(type(lr).__name__))
        self.lr = lr
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not a valid optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        if not isinstance(criterion, nn.modules.loss._Loss):
            raise TypeError('{} is not a valid criterion'.format(type(criterion).__name__))
        self.criterion = criterion
        if not isinstance(evaluator, pl.utils.Evaluator):
            raise TypeError('{} is not a valid pytorchlab.Evaluator'.format(type(evaluator).__name__))
        self.evaluator = evaluator
        if not isinstance(dataloader, pl.dataloaders.Dataloader):
            raise TypeError('{} is not a valid type pytorchlab.Dataloader'.format(type(dataloader).__name__))
        self.dataloader = dataloader
        self.on_cuda = False

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


class ImageClassification(Experiment):
    """Image Classification Design Pattern
    """
    def __init__(self, **kwargs):
        super(ImageClassification, self).__init__(**kwargs)
        self.saver = pl.utils.BestMetricSaver('validation', 'RunningAccuracy', self.experiment_name)

    def _step(self, input, target, is_training=False):
        output = self.model(input)
        loss = self.criterion(output, target)
        self.evaluator.update(output, target)
        self.saver.loss = loss.item()
        if self.model.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.saver.step += 1
        print(self.evaluator.state[self.evaluator.metric_set])
        return output

    def _epoch(self, data_iter):
        self.evaluator.metric_set = data_iter.metric_set
        for input, target in data_iter:
            if self.on_cuda:
                input = input.cuda()
                target = target.cuda()
            output = self._step(input, target)

    def train(self, num_epochs):
        self.model.train()
        train_iter = self.dataloader.train_iter()
        for n in range(num_epochs):
            self._epoch(train_iter)
            self.saver.epoch += 1
            self.saver.model_state_dict = self._get_model_state()
            self.saver.optimizer_state_dict = self.optimizer.state_dict()
            self.saver.save()

    def train_and_validate(self, num_epochs):
        self.model.train()
        train_iter = self.dataloader.train_iter()
        val_iter = self.dataloader.val_iter()
        for n in range(num_epochs):
            self._epoch(train_iter)
            self.saver.epoch += 1
            self.model.eval()
            self._epoch(val_iter)
            self.saver.model_state_dict = self._get_model_state()
            self.saver.optimizer_state_dict = self.optimizer.state_dict()
            self.saver.save(metric_evaluator = self.evaluator)
            self.evaluator.reset()

    def test(self):
        self.model.eval()
        test_iter = self.dataloader.test_iter()
        self._epoch(test_iter)
