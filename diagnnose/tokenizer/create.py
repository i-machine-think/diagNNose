import os
from typing import Dict, List, Union

from transformers import AutoTokenizer, PreTrainedTokenizer

from .c2i import C2I
from .w2i import W2I


def token_to_index(path: str) -> Dict[str, int]:
    """Reads in a newline-separated file of tokenizer entries.

    Parameters
    ----------
    path : str
        Path to a vocabulary file.

    Returns
    -------
    w2i : Dict[str, int]
        Dictionary mapping a token string to its index.
    """
    with open(os.path.expanduser(path), encoding="ISO-8859-1") as f:
        w2i = {token.strip(): idx for idx, token in enumerate(f)}

    return w2i


def create_tokenizer(path: str, notify_unk: bool = False) -> PreTrainedTokenizer:
    """Creates a tokenizer from a path.

    A LSTM tokenizer is defined as a file with an entry at each line,
    and `path` should point towards that file.

    A Transformer tokenizer is defined by its model name, and is
    imported using the AutoTokenizer class.

    Parameters
    ----------
    path : str
        Either the path towards a vocabulary file, or the model name
        of a Huggingface Transformer.
    notify_unk : bool, optional
        Optional toggle to notify a user if a token is not present in
        the vocabulary of the tokenizer. Defaults to False.

    Returns
    -------
    tokenizer : PreTrainedTokenizer
        The instantiated tokenizer that maps tokens to their indices.
    """
    if os.path.exists(os.path.expanduser(path)):
        # Word-based vocabulary, used by older LSTM models
        vocab = W2I(token_to_index(path), notify_unk=notify_unk)

        tokenizer = PreTrainedTokenizer()

        tokenizer.added_tokens_encoder = vocab
        tokenizer.added_tokens_decoder = {idx: w for w, idx in vocab.items()}
        tokenizer.vocab = tokenizer.added_tokens_encoder
        tokenizer.ids_to_tokens = tokenizer.added_tokens_decoder

        tokenizer.unk_token = vocab.unk_token
        tokenizer.eos_token = vocab.eos_token
        tokenizer.pad_token = vocab.pad_token

        tokenizer._tokenize = lambda s: s.split(" ")
        tokenizer._convert_token_to_id = lambda w: vocab[w]

        return tokenizer

    # Subword-based vocabulary, used by Transformer models
    tokenizer = AutoTokenizer.from_pretrained(path)
    if hasattr(tokenizer, "encoder"):
        # GPT-2 & Roberta use a different attribute for the underlying vocab dictionary.
        encoder: Dict[str, int] = getattr(tokenizer, "encoder")
        tokenizer.vocab = W2I(encoder, unk_token=tokenizer.unk_token)
        tokenizer.ids_to_tokens = tokenizer.decoder

    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.unk_token

    return tokenizer


def create_char_vocab(corpus_path: Union[str, List[str]]) -> C2I:
    return C2I(token_to_index(corpus_path))
