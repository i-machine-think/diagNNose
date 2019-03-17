from dataclasses import dataclass
from typing import Any, Dict, List, Optional

Labels = List[int]
Sentence = List[str]


@dataclass
class CorpusSentence:
    """Class that contains a sentence and, optionally, a list of labels.

    Other sentence info is all stored in the `misc_info` dictionary.
    """
    sen: Sentence
    labels: Optional[Labels]
    misc_info: Dict[str, Any]

    def __len__(self) -> int:
        return len(self.sen)

    def validate(self) -> None:
        if self.labels is None:
            return
        sen_len = len(self.sen)
        labels_len = len(self.labels)
        assert sen_len == labels_len, \
            f'Length mismatch between sentence and labels, ' \
            f'{sen_len} (sen) vs. {labels_len} (labels).'


Corpus = Dict[int, CorpusSentence]
