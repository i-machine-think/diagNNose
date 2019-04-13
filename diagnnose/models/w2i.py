from typing import Any, Dict
from warnings import warn


class W2I(dict):
    """ Provides vocab functionality mapping words to indices.

    Non-existing tokens are mapped to the id of an unk token that should
    be present in the vocab file.
    """
    def __init__(self, w2i: Dict[str, int], unk_token: str = '<unk>') -> None:
        if unk_token not in w2i:
            warn('Unk token not present, will be appended to dictionary')
            w2i[unk_token] = len(w2i)

        self.unk_idx = w2i[unk_token]
        super().__init__(w2i)

    def __missing__(self, key: Any) -> Any:
        return self.unk_idx

