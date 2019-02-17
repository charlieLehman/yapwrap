import torch
from torch import nn
from yapwrap.dataloaders import MNIST, FASHION_MNIST
from yapwrap.experiments import ImageClassification
from yapwrap.utils import ImageClassificationEvaluator
from yapwrap.models import MNIST_ConvNet, FASHION_MNIST_ConvNet
import inspect

# Training Data
dataloader = MNIST()
# dataloader = FASHION_MNIST()

# Models to Compare
mnist_net = MNIST_ConvNet()
# fmnist_net = FASHION_MNIST_ConvNet()
models = [mnist_net]

# Evaluation Criterion
evaluator = ImageClassificationEvaluator(dataloader.num_classes)

# Run both experiments
lr = 0.1
num_epochs = 300
for model in models:
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [100,200], 0.1)
    criterion=nn.CrossEntropyLoss()
    kwargs = {'model':model, 'lr':lr, 'lr_scheduler': lr_scheduler, 'optimizer':optimizer, 'criterion':criterion, 'dataloader':dataloader, 'evaluator':evaluator}
    exp = ImageClassification(**kwargs).cuda()
    exp.train_and_validate(300)
# exp.test()
