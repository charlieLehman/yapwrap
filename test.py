import torch
from torch import nn
from yapwrap.dataloaders import CIFAR10
from yapwrap.experiments import ImageClassification
from yapwrap.utils import ImageClassificationEvaluator
from yapwrap.models import TinyResNet18, ComplementConstraint
from torchvision.models import resnet18
import inspect

# Training Data
dataloader = CIFAR10()

# Models to Compare
trn = TinyResNet18(dataloader.num_classes)
trn_cc = ComplementConstraint(trn)
models = [trn, trn_cc]

# Evaluation Criterion
evaluator = ImageClassificationEvaluator(dataloader.num_classes)

# Run both experiments
lr = 0.1
num_epochs = 300
for model in models:
    optimizer = torch.optim.Adam(model.parameters())
    criterion=nn.CrossEntropyLoss()
    kwargs = {'model':model, 'lr':lr, 'optimizer':optimizer, 'criterion':criterion, 'dataloader':dataloader, 'evaluator':evaluator}
    exp = ImageClassification(**kwargs).cuda()
    exp.train_and_validate(300)
# exp.test()


