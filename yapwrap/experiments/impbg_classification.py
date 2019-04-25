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
from yapwrap.experiments import ImageClassification, OutOfDistribution
from yapwrap.modules import *
import torch
from torch import nn
import os
import glob
from tqdm import tqdm
import requests
from io import BytesIO
from PIL import Image

class ImpBGClassification(ImageClassification):
    """Image Classification with the Implicit Background Constraint Design Pattern
    """
    def __init__(self, config=None, experiment_name=None, experiment_number=None, cuda=False):
        super(ImpBGClassification, self).__init__(config, experiment_name, experiment_number, cuda)
        self.impbgloss = ImplicitAttentionLoss()

    def _step(self, input, target, is_training=False):
        seg, attn, impattn, pred = self.model(input)
        crit_loss = self.criterion(pred, target)
        impbg_loss = self.impbgloss(attn, impattn)
        loss = crit_loss + impbg_loss
        eval_update = {'metrics':(pred, target),
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
            if self.visualize_every_n_step is not None and self.max_visualize_batch is not None:
                if self.evaluator.step % self.visualize_every_n_step == 0:
                    viz = getattr(self.model.module,'visualize', None)
                    if callable(viz) and self.make_logs:
                        _inp = input[:self.max_visualize_batch]
                        _seg = seg[:self.max_visualize_batch]
                        _attn = attn[:self.max_visualize_batch]
                        _impattn= impattn[:self.max_visualize_batch]
                        _pred= pred[:self.max_visualize_batch]
                        self.logger.summarize_images(viz(_inp,_seg,_attn,_impattn,_pred), self.dataloader.name, self.evaluator.step)
        return pred


class ImpBGOOD(OutOfDistribution):
    """OOD Image Classification with the Implicit Background Constraint Design Pattern
    """
    def __init__(self, config=None, experiment_name=None, experiment_number=None, cuda=False):
        super(ImpBGOOD, self).__init__(config, experiment_name, experiment_number, cuda)

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
                    seg, attn, px_log = self.model(input)
                    output = seg.sum((-2,-1))/attn.sum((-2,-1))
                self.evaluator.ood_update(output, target)
                tbar.set_description(data_iter.name)
            self.evaluator.ood_run(name)
            viz = getattr(self.model.module,'visualize', None)
            if callable(viz) and self.make_logs:
                self.logger.summarize_images(viz(input), name, self.evaluator.step)
        if self.make_logs:
            self.logger.summarize_scalars(self.evaluator)
        self.evaluator.metric_set = _metric_set

    def _step(self, input, target, is_training=False):
        seg, attn, px_log = self.model(input)
        impbg = 1-torch.sigmoid(-torch.logsumexp(px_log,1, keepdim=True))
        output = seg.sum((-2,-1))/attn.sum((-2,-1))
        _attn = attn.clone().detach().requires_grad_(False)
        impbg_loss = - (_attn*torch.log(torch.clamp(impbg, 1e-5,1))+(1-_attn)*torch.log(torch.clamp((1-impbg),1e-5,1))).mean()
        loss = self.criterion(output, target) + impbg_loss
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
