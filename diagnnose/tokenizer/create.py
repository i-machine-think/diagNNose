import os
from typing import Dict, List, Union

from transformers import AutoTokenizer

from diagnnose.tokenizer.tokenizer import Tokenizer

from .c2i import C2I
from .w2i import W2I


def token_to_index(tokenizer: str) -> Dict[str, int]:
    with open(os.path.expanduser(tokenizer), encoding="ISO-8859-1") as f:
        w2i = {token.strip(): idx for idx, token in enumerate(f)}

    return w2i


def create_tokenizer(
    tokenizer: str, notify_unk: bool = False, to_lower: bool = False
) -> Tokenizer:
    if os.path.exists(os.path.expanduser(tokenizer)):
        return W2I(token_to_index(tokenizer), notify_unk=notify_unk)

    return AutoTokenizer.from_pretrained(tokenizer, do_lower_case=to_lower)


def create_char_vocab(corpus_path: Union[str, List[str]]) -> C2I:
    return C2I(token_to_index(corpus_path))
