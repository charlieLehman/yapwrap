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
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "yapwrap",
    version = "0.0.1",
    author = "Charlie Lehman",
    author_email = "charlie.k.lehman@gmail.com",
    description = ("Yet Another PyTorch Wrapper"),
    license = "Apache",
    keywords = "pytorch research deepleanring machinelearning",
    url = "https://github.gatech.edu/clehman31/yapwrap",
    packages=['yapwrap',
              'yapwrap.experiments',
              'yapwrap.utils',
              'yapwrap.dataloaders',
              'yapwrap.models'],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: Apache",
    ],
    install_requires=[
        "torch",
        "torchvision",
        "scikit-learn",
        "scikit-image",
        "pandas",
        "tqdm",
        "tensorboardX",
],
     dependency_links=[
    ]
)

