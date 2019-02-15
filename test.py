import torch
from torch import nn
from yapwrap.dataloaders import CIFAR10
from yapwrap.experiments import ImageClassification
from yapwrap.utils import ImageClassificationEvaluator
from yapwrap.models import TinyResNet18, ComplementConstraint
from torchvision.models import resnet18
import inspect

dataloader = CIFAR10()
trn = TinyResNet18(dataloader.num_classes)
trn_cc = ComplementConstraint(trn)
models = [trn, trn_cc]

evaluator = ImageClassificationEvaluator(dataloader.num_classes)

lr = 0.1
num_epochs = 300
for model in models:
    optimizer = torch.optim.SGD(model.parameters(), lr = lr , momentum=0.9, weight_decay=1e-4, nesterov=False)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    criterion=nn.CrossEntropyLoss()
    kwargs = {'model':model, 'lr':lr, 'optimizer':optimizer, 'lr_scheduler':lr_scheduler, 'criterion':criterion, 'dataloader':dataloader, 'evaluator':evaluator}
    exp = ImageClassification(**kwargs).cuda()
    exp.train_and_validate(300)
# exp.test()


