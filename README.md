# Yet Another PyTorch Wrapper

## Introduction
I've consolidated a bunch of cool snippets of PyTorch code into something that helps me manage my experiments.  The general philosophy is that an experiment for vision tasks that uses Deep Learning may involve many models and datasets that each have individual nuance necessary to visualize, measure, resume, test, etc.  In order to perform comparative analysis on a large number of metrics without knowing the models and datasets beforehand I wanted a way of constructing the components in a "full-featured" way.  Hence, this wrapper encourages that Models come with methods for visualization, Experiments come with standard measures, visualizations, loggers, and saver, and Datasets come with structure that allow for them to support the former.
## Getting Started
The extent of the features are demonstrated in the Out-of-distribution experiment, as this uses a variety of models and datasets.
## Features
Experiment
Evaluator
Model
Saver
Logger
## TODO
Write a better README
Run from args
Reproduce from an experiment log
## Requirements

