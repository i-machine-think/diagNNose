from typing import BinaryIO, Callable, Dict, List, Tuple, Union

from numpy import ndarray
from torch import Tensor
from torchtext.data import Example

# TENSOR DICTS
ActivationName = Tuple[int, str]  # (layer, name)
ActivationNames = List[ActivationName]

ActivationFiles = Dict[ActivationName, BinaryIO]

# Maps a layer index to a tensor
LayeredTensors = Dict[int, Tensor]
# Maps an arbitrary string to a tensor
NamedTensors = Dict[str, Tensor]

ActivationDict = Dict[ActivationName, Tensor]

# EXTRACTION
# token index, corpus item -> bool
SelectionFunc = Callable[[int, Example], bool]

# [(start, stop)]
ActivationRanges = List[Tuple[int, int]]

RemoveCallback = Callable[[], None]


# INDEXING
# Activation indexing, as done in ActivationReader
ActivationIndex = Union[int, slice, List[int], ndarray, Tensor]

ActivationKey = Union[ActivationIndex, Tuple[ActivationIndex, ActivationName]]


# DECOMPOSITIONS
Decompositions = Dict[int, NamedTensors]
