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
        self.model_original = model
        self.rnn_type = model.rnn_type
        self.nhid = model.nhid
        self.nlayers = model.nlayers
        self.diagnostic_layer = None

    def set_model(self, model, n_layer):
        """
        Crop model to given layer and set as attribute
        to DiagnosticClassifier.
        """
        layers = list(model.children())

        # add padding to embedding
        emb_weights = model.state_dict()['encoder.weight']
        pad = torch.zeros(1, emb_weights.size(1))
        new_emb_weights = torch.cat((emb_weights, pad), 0)
        new_embs = nn.Embedding(emb_weights.size(0)+1, emb_weights.size(1))
        new_embs.weight.data.copy_(new_emb_weights)

        all_layers = layers[:0]+[new_embs]+layers[2:n_layer]
        model = nn.Sequential(*all_layers)

        # freeze model weights
        for param in model.parameters():
            param.requires_grad = False

        return model

    def init_hidden(self, batch_size):
        weight = next(self.model_original.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, batch_size, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, batch_size, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())


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

        for epoch in range(n_epochs):

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
            inputs, input_lengths = getattr(batch, 'sentences')
            targets, _ = getattr(batch, 'targets')      # TODO no idea what this second var is supposed to contain...

            # pack inputs
            # inputs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths)

            # run model
            hidden = self.init_hidden(inputs.size(0))
            self.diagnostic_layer.zero_grad()
            # print type(inputs)

            # print inputs.size()
            # print inputs.size(0)
            # print hidden[0].size(), hidden[1].size()
            # raw_input()
            # hidden = self.model(inputs, hidden)
            layer_output, hidden = self.model(inputs)

            # print hidden
            # raw_input()
            # print hidden2
            # raw_input()

            output = self.diagnostic_layer(layer_output)

            # compute loss and do backward pass
            loss = self.criterion(output, targets)
            loss.backward()
            total_loss += loss.data

            iteration+=1
            no_batch+=1

        return total_loss/no_batch, iteration

