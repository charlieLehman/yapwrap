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
from yapwrap.modules import *
import torch
from torch import nn
import os
import glob
from tqdm import tqdm

class LIDChallengeTask1(ImageClassification):
    """LID Challenge Task 1
    """
    def __init__(self, config=None, experiment_name=None, experiment_number=None, cuda=False):
        super(LIDChallengeTask1, self).__init__(config, experiment_name, experiment_number, cuda)
        self.impbgloss = ImplicitAttentionLoss()

    def _step(self, input, target, is_training=False):
        out, attn, impattn, pred = self.model(input)
        if self.model.training:
            output = pred
        else:
            output = out
        crit_loss = self.criterion(output, target)
        impbg_loss = self.impbgloss(attn, impattn)
        loss = crit_loss + impbg_loss
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
            if self.visualize_every_n_step is not None:
                if self.evaluator.step % self.visualize_every_n_step == 0:
                    viz = getattr(self.model.module,'visualize', None)
                    if callable(viz) and self.make_logs:
                        self.logger.summarize_images(viz(input, target), self.dataloader.name, self.evaluator.step)
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
        if callable(viz) and self.make_logs and self.visualize_every_epoch:
            _input = input[:self.max_visualize_batch] if self.max_visualize_batch is not None else input
            _target = target[:self.max_visualize_batch] if self.max_visualize_batch is not None else input
            self.logger.summarize_images(viz(_input, _target), self.dataloader.name, self.evaluator.step)
