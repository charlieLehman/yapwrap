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

from torch.nn import functional as F
from torch import nn
import torch
import math

class CosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, last_epoch=-1, min_lr_lambda=lambda x:1e-6/x):
        self.T_max = T_max
        self.min_lrs = [min_lr_lambda(pg['lr']) for pg in optimizer.param_groups]
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [min_lr + (base_lr - min_lr)*0.5*(1+math.cos(math.pi*self.last_epoch/self.T_max))
                for min_lr, base_lr in zip(self.min_lrs, self.base_lrs)]

class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    """Sets the learning rate of each parameter group to the initial lr
    decayed by gamma every step_size epochs. When last_epoch=-1, sets
    initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Ending epoch.
        beta (float): Multiplicative factor of learning rate decay.
            Default: 0.9.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self, optimizer, T_max, beta=0.9, last_epoch=-1):
        self.T_max = T_max
        self.beta = beta
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (1.0-self.last_epoch / self.T_max)**0.9
                for base_lr in self.base_lrs]
