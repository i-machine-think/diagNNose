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


Corpus = Dict[int, CorpusSentence]
