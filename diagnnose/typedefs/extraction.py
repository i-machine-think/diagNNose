from typing import Callable, Dict, Tuple

from torchtext.data import Batch


# w position, current w, batch index, total batch -> bool
SelectFunc = Callable[[int, str, int, Batch], bool]

Range = Tuple[int, int]
ActivationRanges = Dict[int, Range]
