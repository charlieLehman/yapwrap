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
from yapwrap.experiments import Experiment
import torch
from torch import nn
import os
import glob
from tqdm import tqdm

class ImageClassification(Experiment):
    """Image Classification Design Pattern
    """
    def __init__(self, config=None, experiment_name=None, experiment_number=None):
        super(ImageClassification, self).__init__(config, experiment_name, experiment_number)

    def _step(self, input, target, is_training=False):
        output = self.model(input)
        loss = self.criterion(output, target)
        eval_update = {'metrics':(output, target),
                        'loss':loss.item(),
                        'criterion':str(self.criterion)}
        self.evaluator.update(**eval_update)
        if self.model.training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.saver.step += 1
            self.evaluator.step += 1
            self.saver.loss = loss.item()
            self.logger.summarize_scalars(self.evaluator)
        return output

    def _epoch(self, data_iter):
        self.evaluator.metric_set = data_iter.metric_set
        tbar = tqdm(data_iter)
        for input, target in tbar:
            if self.on_cuda:
                input = input.cuda()
                target = target.cuda()
            output = self._step(input, target)
            tbar.set_description(self.evaluator.tbar_desc(self.saver.epoch))
        viz = getattr(self.model.module,'visualize', None)
        if callable(viz):
            self.logger.summarize_images(viz(input), self.dataloader.name, self.evaluator.step)

    def train(self, num_epochs, validate=True):
        train_iter = self.dataloader.train_iter()
        if validate:
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
            self.saver.epoch += 1

            if validate:
                self.model.eval()
                with torch.no_grad():
                    self._epoch(val_iter)

            self.logger.summarize_scalars(self.evaluator)
            self.saver.model_state_dict = self._get_model_state()
            self.saver.optimizer_state_dict = self.optimizer.state_dict()
            self.saver.save(metric_evaluator = self.evaluator)

    def test(self):
        self.model.eval()
        test_iter = self.dataloader.test_iter()
        self._epoch(test_iter)
