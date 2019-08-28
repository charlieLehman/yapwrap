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
from yapwrap.experiments import Experiment, get_experiment_dir
import torch
from torch import nn
import os
import glob
from tqdm import tqdm
import requests
from io import BytesIO
from PIL import Image

def Runner(experiment_name, experiment_number, **kwargs):
        experiment_dir = get_experiment_dir(experiment_name, experiment_number)
        saver_ = yapwrap.loggers.Saver(experiment_name, experiment_dir)
        config = saver_.load_config()
        state = torch.load(os.path.join(experiment_dir, 'checkpoint.pth.tar'))
        model_ = kwargs.get('model', None)
        if model_ is None:
            model_ = getattr(yapwrap.models, config['model']['class'])
        model = model_(**config['model']['params'])
        model.load_state_dict(state['model_state_dict'])
        if config.get('cuda', False):
            model = nn.DataParallel(model)
            model.cuda()
            model.name = model.module.name
        return model, config

