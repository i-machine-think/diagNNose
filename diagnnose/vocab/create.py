import glob
import os
from collections import OrderedDict
from typing import Dict, List, Union

from torchtext.vocab import Vocab

from diagnnose.typedefs.corpus import Corpus

from .c2i import C2I
from .w2i import W2I


def create_w2i_dict(corpus_path: Union[str, List[str]]) -> Dict[str, int]:
    if isinstance(corpus_path, str):
        corpus_path = glob.glob(corpus_path)

    corpus_tokens: OrderedDict = OrderedDict()
    for path in corpus_path:
        with open(os.path.expanduser(path)) as cf:
            # Note that the first corpus column is considered to be the sentence here
            corpus_tokens.update(
                (w.strip(), None)
                for l in cf.readlines()
                for w in l.split("\t")[0].split(" ")
            )

    w2i = {w: i for i, w in enumerate(corpus_tokens)}

    return w2i


def create_vocab(corpus_path: Union[str, List[str]]) -> W2I:
    return W2I(create_w2i_dict(corpus_path))


def create_char_vocab(corpus_path: Union[str, List[str]]) -> C2I:
    return C2I(create_w2i_dict(corpus_path))


def attach_vocab(corpus: Corpus, vocab_path: str, sen_column: str = "sen") -> None:
    vocab = create_vocab(vocab_path)

    corpus.fields[sen_column].vocab = Vocab({}, specials=[])
    corpus.fields[sen_column].vocab.stoi = vocab
    corpus.fields[sen_column].vocab.itos = list(vocab.keys())

    corpus.vocab = corpus.fields[sen_column].vocab
