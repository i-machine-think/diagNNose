import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torchtext
from torchtext.vocab import Vocab
import matplotlib.pyplot as plt
import numpy

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


    def add_linear(self, layer=None):
        """
        Add a linear diagnostic classifier. If no file is provided,
        initialise randomly, otherwise, add the layer which is given
        by the layer parameter.
        """
        if layer is not None:
            l = open(layer, 'rb')
            self.diagnostic_layer = torch.load(l)
        else:
            # TODO check here
            self.diagnostic_layer = nn.Linear(self.model[-1].hidden_size, 1)

        self.criterion = torch.nn.MSELoss()

    def evaluate(self, data, batch_size, criterion, device):
        """
        Evaluate model.
        """
        total_loss = 0
        total_loss2 = 0
        batch_no = 0
        no_items = 0

        criterion2=self.get_criterion(criterion, average=True)
        criterion=self.get_criterion(criterion, average=False)

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

            # run model
            
            layer_output, hidden = self.model(inputs)

            output = self.diagnostic_layer(layer_output)

            # create mask to select non-padding values
            non_padding = targets.ne(-1)

            masked_targets = torch.masked_select(targets, non_padding)
            masked_outputs = torch.masked_select(output.squeeze(), non_padding)
            batch_items = masked_outputs.size(0)

            # print masked_targets.size(0), non_padding.long().sum().data[0]

            # compute loss
            loss = criterion(masked_outputs, masked_targets)
            loss2 = criterion2(masked_outputs, masked_targets)
            total_loss += loss.data
            total_loss2 += loss2.data

            no_items+=batch_items
            batch_no+=1

        # print "loss size average", (total_loss2/batch_no)[0]
        # print "total loss not size average", (total_loss/no_items)[0]

        return total_loss/no_items

    def get_criterion(self, criterion, average):
        """
        Create a pytorch criterion based on string
        annotation of criterion
        """
        if criterion == 'mse':
            criterion = torch.nn.MSELoss(size_average=average)
        elif criterion == 'mae':
            criterion = torch.nn.L1Loss(size_average=average)
        else:
            raise ValueError("Criterion not supported")

        return criterion


    def diagnose(self, train_data, val_data, n_epochs, batch_size, print_every=50):

        if not self.diagnostic_layer:
            raise ValueError("Diagnostic layer not set")

        self.optimizer = optim.Adam(self.diagnostic_layer.parameters(), lr=0.001)
        total_loss = 0
        iteration = 0

        device = None if torch.cuda.is_available() else -1

        for epoch in range(n_epochs):

            loss, iteration = self.train_epoch(train_data, batch_size, iteration, device, print_every)
            train_mse = self.evaluate(train_data, batch_size=500, criterion='mse', device=device)
            train_mae = self.evaluate(train_data, batch_size=500, criterion='mae', device=device)
            train_mae = self.evaluate(train_data, batch_size=500, criterion='mae', device=device)
            val_mse = self.evaluate(val_data, batch_size=500, criterion='mse', device=device)
            val_mae = self.evaluate(val_data, batch_size=500, criterion='mae', device=device)
            print("\nEpoch %i: loss: %f, mse train: %f, mae train: %f mse val: %f mae val: %f" % (epoch, loss, train_mse, train_mae, val_mse, val_mae))

            total_loss += loss

        return loss/n_epochs

    def train_epoch(self, data, batch_size, iteration, device, print_every):

        total_loss = 0
        no_batch = 0
        no_items = 0

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
            layer_output, hidden = self.model(inputs)

            output = self.diagnostic_layer(layer_output)

            # create mask to select non-padding values
            non_padding = targets.ne(-1)

            masked_targets = torch.masked_select(targets, non_padding)
            masked_outputs = torch.masked_select(output.squeeze(), non_padding)

            # compute loss and do backward pass
            loss = self.criterion(masked_outputs, masked_targets)
            total_loss += loss.data
            loss.backward()
            self.optimizer.step()
            self.diagnostic_layer.zero_grad()

            iteration+=1
            no_batch+=1

            if iteration % print_every == 0:
                print("Batch loss after iteration %i: %f" % (iteration, total_loss/no_batch))

        return total_loss/no_batch, iteration

    def plot_predictions(self, data):

        vocab = data.fields['sentences'].vocab.itos

        # generate batch iterator
        batch_iterator = torchtext.data.BucketIterator(
                dataset=data, batch_size=1,
                device=-1)


        for batch in batch_iterator:
            # get batch data
            indices, _ = getattr(batch, 'sentences')
            target, _ = getattr(batch, 'targets')
            sentence = [vocab[index.data[0]] for index in indices]

            layer_output, hidden = self.model(indices)
            output = self.diagnostic_layer(layer_output).squeeze().squeeze()

            m = target.max().data[0] + 2
            plt.plot(output.data.numpy(), label='model output', linewidth=2.0)
            plt.plot(target.data.numpy(), label='target', linewidth=2.0)
            plt.xticks(numpy.arange(len(sentence)), sentence)
            plt.yticks(numpy.arange(m), numpy.arange(m))
            plt.legend()
            ax = plt.gca()
            ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
            ax.tick_params(axis = 'both', which = 'minor', labelsize = 16)
            plt.show()
            if raw_input() != '':
                return


    def save(self, name):
        f = open(name, 'wb')
        torch.save(self.diagnostic_layer, f)

