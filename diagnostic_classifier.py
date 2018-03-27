import torch
import torch.nn as nn
from torch.autograd import Variable
import torchtext
from torchtext.vocab import Vocab

# import os
import math

class DiagnosticClassifier():
    """
    Allows to 'diagnose' a sequential model by training additional
    classifiers to predict information from the hidden state
    space.
    """
    def __init__(self, model, n_layer):
        self.model = self.set_model(model, n_layer)
        self.diagnostic_layer = None

    def set_model(self, model, n_layer):
        """
        Crop model to given layer and set as attribute
        to DiagnosticClassifier.
        """
        up_to_layer = list(model.children())[:n_layer]
        model = nn.Sequential(*up_to_layer)

        # freeze model weights
        for param in model.parameters():
            param.requires_grad = False

        return model

    def add_linear(self):
        """
        Add a linear diagnostic classifier.
        """
        self.diagnostic_layer = nn.Linear(self.model[-1].hidden_size, 1)
        self.criterion = torch.nn.MSELoss()

    def diagnose(self, data, n_epochs, batch_size, print_every=50):

        if not self.diagnostic_layer:
            raise ValueError("Diagnostic layer not set")

        total_loss = 0
        iteration = 0

        device = None if torch.cuda.is_available() else -1

        for epoch in xrange(n_epochs):

            loss, iteration = self.train_epoch(data, batch_size, iteration, device)
            print("Loss after epoch %i: %f" % (epoch, loss))

            total_loss += loss

        return loss/n_epochs

    def train_epoch(self, data, batch_size, iteration, device):

        total_loss = 0
        no_batch = 0

        # generate batch iterator
        batch_iterator = torchtext.data.BucketIterator(
                dataset=data, batch_size=batch_size,
                sort=False, sort_within_batch=True,
                sort_key=lambda x: len(x.sentences),
                device=device, repeat=False)

        for batch in batch_iterator:
            # get batch data
            inputs, input_lengths = getattr(batch, sentences)
            targets = getattr(batch, targets)

            # run model
            hidden = model.init_hidden(eval_batch_size)
            model.zero_grad()
            output = model(inputs, hidden)

            # compute loss and do backward pass
            loss = self.criterion(output, targets)
            loss.backward()
            total_loss += loss.data

            iteration+=1
            no_batch+=1

        return total_loss.data/no_batch, iteration

