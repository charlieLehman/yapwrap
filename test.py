import torch
from torch import nn
from pytorchlab.dataloaders import CIFAR10
from pytorchlab.experiments import ImageClassification
from pytorchlab.utils import ImageClassificationEvaluator
from pytorchlab.models import TinyResNet18
from torchvision.models import resnet18
import inspect

def serialize(obj):
    if isinstance(obj, torch.optim.Optimizer):
        serial = repr(obj)
        return serial
    return obj.__dict__

dataloader = CIFAR10()
model = TinyResNet18(dataloader.num_classes)
evaluator = ImageClassificationEvaluator(dataloader.num_classes)

lr = 1.1
optimizer = torch.optim.SGD(model.parameters(), lr = lr , momentum=0.9, weight_decay=1e-4, nesterov=False)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, mode='max', threshold=.01, patience=2)
criterion=nn.CrossEntropyLoss()
kwargs = {'model':model, 'lr':lr, 'optimizer':optimizer, 'lr_scheduler':lr_scheduler, 'criterion':criterion, 'dataloader':dataloader, 'evaluator':evaluator}

exp = ImageClassification(**kwargs).cuda()
exp.train_and_validate(2)
# exp.test()


