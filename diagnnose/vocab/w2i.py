from typing import Dict


class W2I(dict):
    """ Provides vocab functionality mapping words to indices.

    Non-existing tokens are mapped to the id of an unk token that should
    be present in the vocab file.

    Arguments
    ---------
    w2i : Dict[str, int]
        Dictionary that maps strings to indices. This dictionary can be
        created using `create_vocab`.
    unk_token : str, optional
        The unk token to which unknown words will be mapped. Defaults to
        <unk>.
    eos_token : str, optional
        The end-of-sentence token that is used in the corpus. Defaults
        to <eos>.
    """

    def __init__(
        self, w2i: Dict[str, int], unk_token: str = "<unk>", eos_token: str = "<eos>"
    ) -> None:
        if unk_token not in w2i:
            w2i[unk_token] = len(w2i)
        if eos_token not in w2i:
            w2i[eos_token] = len(w2i)

        super().__init__(w2i)

        self.unk_idx = w2i[unk_token]
        self.unk_token = unk_token
        self.eos_token = eos_token

        self.i2w = list(w2i.keys())

    @property
    def w2i(self) -> Dict[str, int]:
        return self

    def __missing__(self, key: str) -> int:
        return self.unk_idx
