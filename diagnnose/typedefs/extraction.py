from typing import Callable, Dict, Tuple

from torchtext.data import Example


# w position, current w, batch item -> bool
SelectFunc = Callable[[int, int, Example], bool]

Range = Tuple[int, int]
ActivationRanges = Dict[int, Range]
