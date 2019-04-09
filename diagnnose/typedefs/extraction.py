from typing import Callable, Dict, Tuple

from .corpus import LabeledSentence

SelectFunc = Callable[[int, str, LabeledSentence], bool]

Range = Tuple[int, int]
ActivationRanges = Dict[int, Range]
