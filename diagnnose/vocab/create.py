import os
from collections import OrderedDict
from typing import List, Set, Union

from .c2i import C2I
from .w2i import W2I


def create_vocab(
    corpus_path: str, create_char_vocab: bool = False
) -> Union[W2I, C2I]:
    with open(os.path.expanduser(corpus_path)) as cf:
        # Note that the first corpus column is considered to be the sentence here
        corpus_tokens = OrderedDict.fromkeys(
            w.strip() for l in cf.readlines() for w in l.split("\t")[0].split(" ")
        )

    w2i = {w: i for i, w in enumerate(corpus_tokens)}

    if create_char_vocab:
        return C2I(w2i)
    else:
        return W2I(w2i)
