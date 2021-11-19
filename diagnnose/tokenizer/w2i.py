from typing import Dict
from warnings import warn


class W2I(dict):
    """Provides vocab functionality mapping words to indices.

    Non-existing tokens are mapped to the id of an unk token that should
    be present in the vocab file.

    Parameters
    ----------
    w2i : Dict[str, int]
        Dictionary that maps strings to indices. This dictionary can be
        created using `create_vocab`.
    unk_token : str, optional
        The unk token to which unknown words will be mapped. Defaults to
        <unk>.
    eos_token : str, optional
        The end-of-sentence token that is used in the corpus. Defaults
        to <eos>.
    notify_unk : bool, optional
        Notify when a requested token is not present in the vocab.
        Defaults to False.
    """

    def __init__(
        self,
        w2i: Dict[str, int],
        unk_token: str = "<unk>",
        eos_token: str = "<eos>",
        pad_token: str = "<pad>",
        notify_unk: bool = False,
    ) -> None:
        super().__init__(w2i)

        if unk_token not in w2i:
            warn(f"Unk token {unk_token} not found in provided vocab.")

        self.unk_token = unk_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        self.unk_idx = w2i.get(unk_token, None)

        self.notify_unk = notify_unk

        self.i2w = list(w2i.keys())

    @property
    def w2i(self) -> Dict[str, int]:
        return self

    def __missing__(self, key: str) -> int:
        if self.notify_unk:
            warn(f"`{key}` is not present in vocab")
        return self.unk_idx
