import os
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = ['<unk>']

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
            self.word2count[word]=1
        else:
            self.word2count[word] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def add_corpus(self, path):
        """Tokenizes a text file and add words
        to the dictionary."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.add_word(word)
        return

    def tokenise_corpus(self, path):
        """ Tokenize file content"""

        # count number of words
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)

        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    try:
                        ids[token] = self.word2idx[word]
                    except KeyError:
                        ids[token] = self.word2idx['<unk>']
                    token += 1

        return ids

    def trim(self, min_count):
        keep = ['<unk>']

        for k, v in self.word2count.items():
            if v >= min_count:
                keep.append(k)

        # reiniialise dictionaries
        self.word2idx = {}
        self.word2count = {}
        self.idx2word = []
        for word in keep:
            self.add_word(word)

    def ntokens(self):
        return len(self.idx2word)
