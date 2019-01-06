from typing import Any, List
from dataclasses import dataclass


Labels = List[int]
Sentence = List[str]


@dataclass
class LabeledSentence:
    """Class that contains a sentence and a list of labels."""
    sen: Sentence
    labels: Labels
    senInfo: Any = None

    def __len__(self) -> int:
        return len(self.sen)

    def validate(self) -> None:
        sen_len = len(self.sen)
        labels_len = len(self.labels)
        assert sen_len == labels_len, \
            f'Length mismatch between sentence and labels, ' \
            f'{sen_len} (sen) vs. {labels_len} (labels).'


LabeledCorpus = List[LabeledSentence]
