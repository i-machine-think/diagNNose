from typing import Union

from torchtext.vocab import Vocab
from transformers import PreTrainedTokenizer

from .w2i import W2I

Tokenizer = Union[PreTrainedTokenizer, Vocab, W2I]
