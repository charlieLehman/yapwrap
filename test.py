import torch
from torch import nn
from yapwrap.dataloaders import CIFAR10
from yapwrap.experiments import ImageClassification
from yapwrap.utils import ImageClassificationEvaluator
from yapwrap.models import TinyResNet18
from torchvision.models import resnet18
import inspect

dataloader = CIFAR10()
model = TinyResNet18(dataloader.num_classes)
evaluator = ImageClassificationEvaluator(dataloader.num_classes)

lr = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr = lr , momentum=0.9, weight_decay=1e-4, nesterov=False)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, mode='max', threshold=.01, patience=2)
criterion=nn.CrossEntropyLoss()
kwargs = {'model':model, 'lr':lr, 'optimizer':optimizer, 'lr_scheduler':lr_scheduler, 'criterion':criterion, 'dataloader':dataloader, 'evaluator':evaluator}

exp = ImageClassification(**kwargs).cuda()
exp.train_and_validate(10)
# exp.test()


