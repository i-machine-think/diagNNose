# coding: utf-8
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import Counter
import torchtext
from torchtext.vocab import Vocab
from diagnostic_classifier import DiagnosticClassifier

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='location of the training data')
parser.add_argument('--model', type=str, help='path to load the model')
parser.add_argument('--vocab', type=str, help='path to load the model dict')
parser.add_argument('--bptt', type=int, default=60, help='max sequence length')
parser.add_argument('--batch_size', type=int, default=24, help='batch size for training')
parser.add_argument('--eval_batch_size', type=int, default=64, help='eval batch size')
parser.add_argument('--cuda', action='store_true')

######################################################################
# Prepare data

try:
    raw_input
except:
    raw_input = input

args = parser.parse_args()
max_len = 20
device = None if torch.cuda.is_available() else -1

def len_filter(example):
    return len(example.sentences) <= max_len and len(example.targets) <= max_len

def tokeniser(text):
    return text.split()

def tokeniser_targets(targets):
    return [float(target) for target in targets.split()]

def preprocessing(seq):
    return seq

def get_vocab(vocab_file):
    vocab = Vocab(Counter())
    words = open(vocab_file, 'rb').read().splitlines()
    pad = vocab.itos[0]
    vocab.itos = []
    for word in words:
        vocab.itos.append(word)
    vocab.itos.append(pad)
    vocab.stoi.update({tok: i for i, tok in enumerate(vocab.itos)})
    return vocab

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)

if __name__ == '__main__':

    args = parser.parse_args()

    # load model
    with open(args.model, 'rb') as f:
        model = torch.load(f, map_location=lambda storage, loc: storage)

    # generate datafields
    sentences = torchtext.data.Field(sequential=True, tokenize=tokeniser, preprocessing=preprocessing, include_lengths=True, use_vocab=True)
    targets = torchtext.data.Field(sequential=True, tokenize=tokeniser_targets, use_vocab=False, include_lengths=True, tensor_type=torch.FloatTensor, pad_token=-1.)

    # generate vocab and attach to data field
    vocab = get_vocab(args.vocab)
    sentences.vocab = vocab
    pad = vocab.stoi['<pad>']

    # generate train and test data and vocabulary
    train_data = torchtext.data.TabularDataset(
        path=args.data, format='tsv',
        fields=[('sentences', sentences), ('targets', targets)],
        filter_pred=len_filter
    )

    # test_data = torchtext.data.TabularDataset(
    #     path=args.test, format='tsv',
    #     fields=[('sentences', sentences), ('targets', targets)],
    #     filter_pred=len_filter
    # )

    # create diagnostic classifier
    dc = DiagnosticClassifier(model, n_layer=-1)

    dc.add_linear()
    
    loss = dc.diagnose(train_data, n_epochs=100, batch_size=10)

    print(loss)

    # test_loss = evaluate(model, dictionary, test_data, criterion, args.eval_batch_size)
    # print('=' * 89)
    # print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        # test_loss, math.exp(test_loss)))
    # print('=' * 89)
