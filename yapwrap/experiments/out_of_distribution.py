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
from yapwrap.experiments import ImageClassification
from yapwrap.utils import OODEvaluator
import torch
from torch import nn
import os
import glob
from tqdm import tqdm
from collections.abc import Iterable
import pandas as pd
import pickle as pkl
import copy


class OutOfDistribution(ImageClassification):
    """Image Classification Design Pattern
    """
    def __init__(self, config=None, experiment_name=None, experiment_number=None, cuda=False):
        super(OutOfDistribution, self).__init__(config, experiment_name, experiment_number, cuda)

        if isinstance(self.config['ood_dataloaders']['class'], str):
            _ood_dataloaders = getattr(yapwrap.dataloaders, self.config['ood_dataloaders']['class'])
            #TODO Fix this to not have instances prior
            dataloader_list = [getattr(yapwrap.dataloaders, x['name'])() for x in self.config['ood_dataloaders']['params']['dataloader_list']]
            self.ood_dataloaders = _ood_dataloaders(dataloader_list, self.dataloader.test_transform,  in_domain_total=len(self.dataloader.test_iter()))
        else:
            self.ood_dataloaders = self.config['ood_dataloaders']['class'](**self.config['ood_dataloaders']['params'], transform=self.dataloader.test_transform, in_domain_total = len(self.dataloader.test_iter()))

        self.ood_iters = [(dataloader.name, dataloader.ood_iter()) for dataloader in self.ood_dataloaders]
        self.experiment_name = '{}_OOD'.format(self.experiment_name)

    def _ood_run(self):
        _metric_set = self.evaluator.metric_set
        self.evaluator.metric_set = 'OOD'
        for name, data_iter in self.ood_iters:
            tbar = tqdm(data_iter)
            for input, target in tbar:
                if self.on_cuda:
                    input = input.cuda()
                    target = target.cuda()
                ood = getattr(self.model.module,'detect_ood', None)
                if callable(ood):
                    output = ood(input)
                else:
                    output = self.model(input)
                self.evaluator.ood_update(output, target)
                tbar.set_description(data_iter.name)
            self.evaluator.ood_run(name)
            viz = getattr(self.model.module,'visualize', None)
            if callable(viz) and self.make_logs:
                self.logger.summarize_images(viz(input), name, self.evaluator.step)
        if self.make_logs:
            self.logger.summarize_scalars(self.evaluator)
        self.evaluator.metric_set = _metric_set

    def train(self, num_epochs):
        train_iter = self.dataloader.train_iter()
        val_iter = self.dataloader.val_iter()
        for n in range(num_epochs):
            self.evaluator.reset()
            self.model.train()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step(n)
            self._epoch(train_iter)
            if self.lr_scheduler is not None:
                for i, lr in enumerate(self.lr_scheduler.get_lr()):
                    self.evaluator.update(**{'values':{'lr_{}'.format(i):lr}})

            self.model.eval()
            with torch.no_grad():
                self._epoch(val_iter)
                self._ood_run()
            self.saver.epoch += 1

            self.logger.summarize_scalars(self.evaluator)
            self.saver.model_state_dict = self._get_model_state()
            self.saver.optimizer_state_dict = self.optimizer.state_dict()
            self.saver.save(metric_evaluator = self.evaluator)

    def _ood_test(self, metrics_path=None):
        evaluator = self.evaluator
        self.model.eval()
        if metrics_path is None:
            self.metrics_path = self.saver.experiment_dir
        else:
            self.metrics_path = metrics_path
        with torch.no_grad():
            self._epoch(self.dataloader.val_iter())
            evaluator.metric_set = 'OOD'
            for name, data_iter in self.ood_iters:
                tbar = tqdm(data_iter)
                for input, target in tbar:
                    if self.on_cuda:
                        input = input.cuda()
                        target = target.cuda()
                    ood = getattr(self.model.module,'detect_ood', None)
                    if callable(ood):
                        output = ood(input)
                    else:
                        output = self.model(input)
                    evaluator.ood_update(output, target)
                    tbar.set_description(data_iter.name)
                evaluator.ood_run(name)
                viz = getattr(self.model.module,'visualize', None)
                if callable(viz):
                    viz_dict = viz(input)

    def test(self, num_trials=5):
        self.make_logs = False
        self.model.eval()
        evst_d = {}
        with torch.no_grad():
            self._epoch(self.dataloader.test_iter())
            for trial in range(num_trials):
                self._ood_run()
                evst_d.update({trial:copy.deepcopy(self.evaluator.state)})

        return evst_d
