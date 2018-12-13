# coding: utf-8

import argparse
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from evaluate import get_dict
import os

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--model', type=str,  default='model.pt', help='model to use')
parser.add_argument('--vocab', type=str, help='path to load the model dict')
parser.add_argument('--bptt', type=int, default=60, help='sequence length')
args = parser.parse_args()


def tokenise(sentence, dictionary):
    words = sentence.split(' ')
    l = len(words)
    assert l <= args.bptt, "sentence too long"
    token = 0
    ids = torch.LongTensor(l)

    for word in words:
        try:
            ids[token] = dictionary[word]
        except KeyError:
            print("%s unknown, replace by <unk>" % word)
            ids[token] = dictionary['<unk>']
        token += 1
    return ids


# softmax function
softmax = nn.LogSoftmax()


def evaluate(model, dictionary, sentence, check_words):
    sentence = sentence.strip()

    # Turn on evaluation mode which disables dropout.
    model.eval()
    ntokens = len(dictionary)
    hidden = model.init_hidden(1)

    test_data = tokenise(sentence, dictionary)
    input_data = Variable(test_data, volatile=False)

    output, hidden = model(input_data.view(-1,1), hidden)
    output_flat = output.view(-1, ntokens)
    logit = output[-1, :]
    sm = softmax(logit).view(ntokens)

    def get_prob(word):
        return sm[dictionary[word]].data[0]

    print('\n'.join(
            ['%s: %f' % (word, get_prob(word)) for word in check_words]
            ))

    return


if __name__ == '__main__':
    # test sentence and words to check
    test_sentence = 'he does like'
    check_words = ['anybody', 'somebody']
    print('\n', test_sentence, '\n')

    # create dictionary with word_ids
    dictionary = get_dict(args.vocab)

    with open(args.model, 'rb') as f:
        model = torch.load(f, map_location=lambda storage, loc: storage)

    evaluate(model, dictionary, test_sentence, check_words)
