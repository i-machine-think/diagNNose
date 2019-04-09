# diagnnose &middot; [![Build Status](https://travis-ci.com/i-machine-think/diagnnose.svg?branch=master)](https://travis-ci.com/i-machine-think/diagnnose)

This library contains a set of modules that can be used to analyse recurrent neural networks.
In particular, it contains functionality to:
- Extracting activations from different types of (language) models
- Running diagnostic classifiers [1] on extracted activations
- Doing intervention studies [2] with language models
- Analysing word embeddings
- Doing dimensionality reduction and plotting state space trajectories of trained models

# Quickstart

Our library is not (yet) officially registered with pip. You can use the library by cloning it and do an editable install with pip:

```
git clone git@github.com:i-machine-think/diagnnose.git $custom_path
pip3 install -e $custom_path
```

We will shortly update this README with explanations for the different scripts provided in the library.


## Requirements

This library runs with Pytorch 1.0. We refer to the [PyTorch website](http://pytorch.org/) to install the right version for your environment.
The preferred version of python is >=3.7.

To install additional requirements, run:

`pip install -r requirements.txt`
