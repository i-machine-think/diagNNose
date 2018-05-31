# coding: utf-8

import argparse
import torch
import torch.nn as nn
# import os
import math

# import model

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='location of the test data')
parser.add_argument('--model', type=str, help='path to load the model')
parser.add_argument('--vocab', type=str, help='path to load the model dict')
parser.add_argument('--bptt', type=int, default=60, help='sequence length')
parser.add_argument('--eval_batch_size', type=int, default=64, help='eval batch size')
parser.add_argument('--cuda', action='store_true')

def get_dict(vocab_file):
    # create list of words in file
    words =  open(vocab_file, 'rb').read().splitlines()
    vocab = dict(zip(words, xrange(len(words))))
    return vocab

def preprocess(data, dictionary):
    """ Preprocess file content replacing words with ids"""

    # count number of words
    with open(data, 'rb') as f:
        tokens = 0
        for line in f:
            words = line.split() + ['<eos>']
            tokens += len(words)

    with open(data, 'r') as f:
        ids = torch.LongTensor(tokens)
        token = 0
        for line in f:
            words = line.split() + ['<eos>']
            for word in words:
                try:
                    ids[token] = dictionary[word]
                except KeyError:
                    ids[token] = dictionary['<unk>']
                token += 1
    return ids

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == tuple:
        return tuple(repackage_hidden(v) for v in h)
    else:
        return h.item()

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate(model, dictionary, data_source, criterion, eval_batch_size):
    # Turn on evaluation mode which disables dropout.
    model.eval()

    # generate batch iterator
    batch_iterator = torchtext.data.BucketIterator(
            dataset=train, batch_size=args.eval_batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=device, repeat=False)

    total_loss = 0
    ntokens = len(dictionary)
    hidden = model.init_hidden(eval_batch_size)

    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)
        output, hidden = model(data, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


if __name__ == '__main__':

    args = parser.parse_args()
    dictionary = get_dict(args.vocab)

    with open(args.model, 'rb') as f:
        model = torch.load(f, map_location=lambda storage, loc: storage)

    criterion = nn.CrossEntropyLoss()

    corpus_test = preprocess(args.data, dictionary)
    test_data = batchify(corpus_test, args.eval_batch_size)
    test_loss = evaluate(model, dictionary, test_data, criterion, args.eval_batch_size)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
