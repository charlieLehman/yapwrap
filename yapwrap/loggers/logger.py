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

import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import numpy as np

class Logger(object):
    def __init__(self, experiment_name, experiment_dir):
        self.experiment_name = experiment_name
        self.experiment_dir = experiment_dir
        self.writer = SummaryWriter(experiment_dir)

    def process_state(self, state, metric_set, step):
        for k, v in state.items():
            summary_name = os.path.join(k, metric_set)
            if np.isscalar(v):
                self.writer.add_scalar(summary_name, v, step)
            if isinstance(v, dict):
                self.process_state(v, metric_set, step)

    def summarize_scalars(self, evaluator):
        state = evaluator.state[evaluator.metric_set]
        self.process_state(state, evaluator.metric_set, evaluator.step)

    def summarize_images(self, images, name, step):
        for k, v in images.items():
            if type(v) is torch.Tensor:
                s = v.shape
                if len(s)==4:
                    v = make_grid(v, int(np.sqrt(s[0])))
                summary_name = os.path.join(name, k)
                self.writer.add_image(summary_name, v, step)
