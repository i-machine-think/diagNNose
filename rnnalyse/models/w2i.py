from typing import Any, Dict, List


class W2I(dict):
    """ Provides vocab functionality mapping words to indices.

    Non-existing tokens are mapped to the id of an unk token that should
    be present in the vocab file.
    """
    def __init__(self, vocab_lines: List[str], unk_token: str = '<unk>') -> None:
        w2i: Dict[str, int] = {w.strip(): i for i, w in enumerate(vocab_lines)}
        self.unk_idx = w2i[unk_token]
        super().__init__(w2i)

    def __missing__(self, key: Any) -> Any:
        return self.unk_idx
