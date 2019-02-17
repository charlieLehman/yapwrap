import torch
from torch import nn
from yapwrap.dataloaders import MNIST
from yapwrap.experiments import ImageClassification
from yapwrap.utils import ImageClassificationEvaluator
from yapwrap.models import MNIST_ConvNet, FASHION_MNIST_ConvNet
import inspect

# Training Data
dataloader = MNIST()

# Models to Compare
mnist_net = MNIST_ConvNet()
fmnist_net = FASHION_MNIST_ConvNet()
models = [mnist_net, fmnist_net]

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
