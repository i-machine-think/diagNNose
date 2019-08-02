import os
from collections import OrderedDict
from typing import Dict

from .c2i import C2I
from .w2i import W2I


def create_w2i_dict(corpus_path: str) -> Dict[str, int]:
    with open(os.path.expanduser(corpus_path)) as cf:
        # Note that the first corpus column is considered to be the sentence here
        corpus_tokens = OrderedDict.fromkeys(
            w.strip() for l in cf.readlines() for w in l.split("\t")[0].split(" ")
        )

    w2i = {w: i for i, w in enumerate(corpus_tokens)}

    return w2i


def create_vocab(corpus_path: str) -> W2I:
    return W2I(create_w2i_dict(corpus_path))


def create_char_vocab(corpus_path: str) -> C2I:
    return C2I(create_w2i_dict(corpus_path))
