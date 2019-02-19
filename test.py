import torch
from torch import nn
from yapwrap.dataloaders import CIFAR10
from yapwrap.experiments import ImageClassification
from yapwrap.utils import ImageClassificationEvaluator
from yapwrap.utils.lr_scheduler import PolyLR
from yapwrap.models import TinyResNet18, ComplementConstraint, TinyResNet50
from torchvision.models import resnet18
import inspect

# Training Data
dataloader = CIFAR10()

# Models to Compare
trn = TinyResNet18(dataloader.num_classes)
trn_cc = ComplementConstraint(TinyResNet18(dataloader.num_classes))
models = [trn, trn_cc]

# Evaluation Criterion

# Run both experiments
lr = 0.1
num_epochs = 10
for model in models:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=False)
    lr_scheduler = PolyLR(optimizer, num_epochs)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100,200], 0.1)
    # optimizer = torch.optim.Adam(model.parameters())
    # lr_scheduler = None
    criterion = nn.CrossEntropyLoss()
    evaluator = ImageClassificationEvaluator(dataloader.num_classes)
    kwargs = {'model':model, 'lr':lr, 'optimizer':optimizer, 'criterion':criterion, 'lr_scheduler':lr_scheduler, 'dataloader':dataloader, 'evaluator':evaluator}
    exp = ImageClassification(**kwargs).cuda()
    exp.train(num_epochs)
# exp.test()
