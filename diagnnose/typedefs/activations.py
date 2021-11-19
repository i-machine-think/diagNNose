from typing import BinaryIO, Callable, Dict, List, Tuple, Union

from torch import Tensor
from torchtext.data import Example

# ACTIVATION DICTS
ActivationName = Tuple[int, str]  # (layer, name)
ActivationNames = List[ActivationName]

ActivationDict = Dict[ActivationName, Tensor]

# LM's layer sizes: (layer, name) -> size
SizeDict = Dict[ActivationName, int]


# EXTRACTION
ActivationFiles = Dict[ActivationName, BinaryIO]

# token index, corpus item -> bool
SelectionFunc = Callable[[int, Example], bool]

# [(start, stop)]
ActivationRanges = List[Tuple[int, int]]

RemoveCallback = Callable[[], None]


# INDEXING
# Activation indexing, as done in ActivationReader
ActivationIndex = Union[int, slice, List[int], Tensor]

ActivationKey = Union[ActivationIndex, Tuple[ActivationIndex, ActivationName]]
