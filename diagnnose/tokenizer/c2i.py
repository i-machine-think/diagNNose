from typing import Any, Dict, Set

from unidecode import unidecode

from .w2i import W2I


class C2I(W2I):
    """Vocabulary containing character-level information.

    Adapted from: https://github.com/tensorflow/models/tree/master/research/lm_1b
    """

    def __init__(
        self, w2i: Dict[str, int], max_word_length: int = 50, **kwargs: Any
    ) -> None:
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
            raise ValueError("Not enough free char ids: %d" % len(free_ids))

        self.eos_char = free_ids[1]  # <end sentence>
        self.bow_char = free_ids[2]  # <begin word>
        self.eow_char = free_ids[3]  # <end word>
        self.pad_char = free_ids[4]  # <padding>

        self._word_char_ids = {}

        for w in self.w2i.keys():
            self._word_char_ids[w] = self._convert_word_to_char_ids(w)

    @property
    def max_word_length(self) -> int:
        return self._max_word_length

    def _convert_word_to_char_ids(self, word: str):
        import numpy as np

        code = np.zeros([self.max_word_length], dtype=np.int32)
        code[:] = ord(self.pad_char)

        if len(word) > self.max_word_length - 2:
            word = word[: self.max_word_length - 2]
        cur_word = self.bow_char + word + self.eow_char
        for j in range(len(cur_word)):
            code[j] = ord(cur_word[j])
        return code.reshape((1, 1, -1))

    def token_to_char_ids(self, token: str):
        if not all(ord(c) < 256 for c in token):
            token = unidecode(token)

        if token in self._word_char_ids:
            char_ids = self._word_char_ids[token]
        else:
            char_ids = self._convert_word_to_char_ids(token)
            self._word_char_ids[token] = char_ids

        return char_ids
