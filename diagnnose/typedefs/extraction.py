from typing import Callable, Dict, Tuple

from .corpus import CorpusSentence

SelectFunc = Callable[[int, str, CorpusSentence], bool]

Range = Tuple[int, int]
ActivationRanges = Dict[int, Range]
