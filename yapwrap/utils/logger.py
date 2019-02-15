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
from dataloaders.utils import decode_seg_map_sequence

class Logger(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        raise NotImplementedError

    def visualize_image(self):
        raise NotImplementedError


class ImageExperimentLogger(Logger):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, id_conf, bg_conf, global_step):
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        grid_id = make_grid(id_conf[:3].clone().cpu().data, 3, normalize=False)
        grid_bg = make_grid(bg_conf[:3].clone().cpu().data, 3, normalize=False)
        writer.add_image('0_Image', grid_image, global_step)
        writer.add_image('1_InDist Conf', grid_id, global_step)
        writer.add_image('2_BG Conf', grid_bg, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('4_Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('3_Groundtruth label', grid_image, global_step)

