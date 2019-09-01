from typing import BinaryIO, Callable, Dict, List, Tuple, Union

from numpy import ndarray
from torch import Tensor, float32, float64
from torchtext.data import Example

# Tensor dtype that will be used in the library. Can be set here to change.
DTYPE = float32

# TENSOR DICTS
ActivationName = Tuple[int, str]  # (layer, name)
ActivationNames = List[ActivationName]

ActivationFiles = Dict[ActivationName, BinaryIO]

# Maps a layer index to a tensor
LayeredTensors = Dict[int, Tensor]
# Maps an arbitrary string to a tensor
NamedTensors = Dict[str, Tensor]

ActivationTensors = Dict[ActivationName, Tensor]
ActivationTensorLists = Dict[ActivationName, List[Tensor]]

# Batch id to ActivationTensors
BatchActivationTensors = Dict[int, Dict[ActivationName, Tensor]]
# Batch id to ActivationTensorLists
BatchActivationTensorLists = Dict[int, ActivationTensorLists]


# EXTRACTION
# sen_id, w position, batch item -> bool
SelectFunc = Callable[[int, int, Example], bool]

Range = Tuple[int, int]
ActivationRanges = Dict[int, Range]


# INDEXING
# Activation indexing, as done in ActivationReader
ActivationIndex = Union[int, slice, List[int], ndarray, Tensor]

IndexType = (
    str
)  # 'pos', 'key' or 'all' TODO: update indextype when moving to python 3.8
ActivationKeyConfig = Dict[str, Union[ActivationName, IndexType]]

ActivationKey = Union[ActivationIndex, Tuple[ActivationIndex, ActivationKeyConfig]]


# DECOMPOSITIONS
Decompositions = Dict[int, NamedTensors]
