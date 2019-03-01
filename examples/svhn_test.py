import torch
from torch import nn
from yapwrap.dataloaders import SVHN
from yapwrap.experiments import ImageClassification
from yapwrap.utils import ImageClassificationEvaluator
from yapwrap.models import AlexNet
import inspect

# Training Data
dataloader = SVHN()

# Models to Compare
alexnet_svhn = AlexNet()
models = [alexnet_svhn]

# Evaluation Criterion


# Run both experiments
lr = 0.01
num_epochs = 10
for model in models:
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [5], 0.1)
    criterion=nn.CrossEntropyLoss()
    evaluator = ImageClassificationEvaluator(dataloader.num_classes)
    kwargs = {'model':model, 'lr':lr, 'lr_scheduler': lr_scheduler, 'optimizer':optimizer, 'criterion':criterion, 'dataloader':dataloader, 'evaluator':evaluator}
    exp = ImageClassification(**kwargs).cuda()
    exp.train_and_validate(num_epochs)
# exp.test()
