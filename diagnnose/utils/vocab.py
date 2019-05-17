import os
from typing import Any, Dict, Set

import numpy as np


class W2I(dict):
    """ Provides vocab functionality mapping words to indices.

    Non-existing tokens are mapped to the id of an unk token that should
    be present in the vocab file.

    Arguments
    ---------
    w2i : Dict[str, int]
        Dictionary that maps strings to indices. This dictionary can be
        created using `create_vocab_from_path`.
    unk_token : str, optional
        The unk token to which unknown words will be mapped. Defaults to
        <unk>.
    eos_token : str, optional
        The end-of-sentence token that is used in the corpus. Defaults
        to <eos>.
    """

    def __init__(self,
                 w2i: Dict[str, int],
                 unk_token: str = '<unk>',
                 eos_token: str = '<eos>') -> None:
        if unk_token not in w2i:
            w2i[unk_token] = len(w2i)
        if eos_token not in w2i:
            w2i[eos_token] = len(w2i)

        self.unk_idx = w2i[unk_token]
        self.unk_token = unk_token
        self.eos_token = eos_token

        super().__init__(w2i)

    @property
    def w2i(self) -> Dict[str, int]:
        return self

    def __missing__(self, key: str) -> int:
        return self.unk_idx


class C2I(W2I):
    """Vocabulary containing character-level information.
    Taken from: https://github.com/tensorflow/models/tree/master/research/lm_1b
    """

    def __init__(self, w2i: Dict[str, int], max_word_length: int = 50, **kwargs: Any) -> None:
        super().__init__(w2i, **kwargs)
        self._max_word_length = max_word_length
        chars_set: Set[str] = set()

        for word in w2i:
            chars_set |= set(word)

        free_ids = []
        for i in range(256):
            if chr(i) in chars_set:
                continue
            free_ids.append(chr(i))

        if len(free_ids) < 5:
            raise ValueError('Not enough free char ids: %d' % len(free_ids))

        self.eos_char = free_ids[1]  # <end sentence>
        self.bow_char = free_ids[2]  # <begin word>
        self.eow_char = free_ids[3]  # <end word>
        self.pad_char = free_ids[4]  # <padding>

        chars_set |= {self.eos_char, self.bow_char, self.eow_char, self.pad_char}

        self._char_set = chars_set
        num_words = len(w2i)

        self._word_char_ids = np.zeros([num_words, max_word_length], dtype=np.int32)

        self.eos_chars = self._convert_word_to_char_ids(self.eos_char)

        for w, i in self.w2i.items():
            self._word_char_ids[i] = self._convert_word_to_char_ids(w)

    @property
    def max_word_length(self) -> int:
        return self._max_word_length

    def _convert_word_to_char_ids(self, word: str) -> np.ndarray:
        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = ord(self.pad_char)

        if len(word) > self.max_word_length - 2:
            word = word[:self.max_word_length - 2]
        cur_word = self.bow_char + word + self.eow_char
        for j in range(len(cur_word)):
            code[j] = ord(cur_word[j])
        return code

    def word_to_char_ids(self, word: str) -> np.ndarray:
        if word in self.w2i:
            return self._word_char_ids[self.w2i[word]]
        else:
            return self._convert_word_to_char_ids(word)


def create_vocab_from_path(vocab_path: str) -> Dict[str, int]:
    with open(os.path.expanduser(vocab_path), 'r') as vf:
        vocab_lines = vf.readlines()

    w2i = {w.strip(): i for i, w in enumerate(vocab_lines)}

    return w2i
