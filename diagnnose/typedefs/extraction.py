from typing import Callable, Dict, Tuple

from torchtext.data import Example


SelectFunc = Callable[[int, str, Example], bool]

Range = Tuple[int, int]
ActivationRanges = Dict[int, Range]
