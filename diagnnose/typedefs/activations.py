from typing import BinaryIO, Callable, Dict, List, Tuple, Union

from numpy import ndarray
from torch import Tensor
from torchtext.data import Example


# TENSOR DICTS
ActivationName = Tuple[int, str]  # (layer, name)
ActivationNames = List[ActivationName]

ActivationFiles = Dict[ActivationName, BinaryIO]

# Maps a layer index to a tensor
LayeredTensorDict = Dict[int, Tensor]

TensorDict = Dict[ActivationName, Tensor]
TensorListDict = Dict[ActivationName, List[Tensor]]

BatchTensorDict = Dict[int, Dict[ActivationName, Tensor]]
BatchTensorListDict = Dict[int, TensorListDict]


# EXTRACTION
# sen_id, w position, batch item -> bool
SelectFunc = Callable[[int, int, Example], bool]

Range = Tuple[int, int]
ActivationRanges = Dict[int, Range]


# INDEXING
# Activation indexing, as done in ActivationReader
ActivationIndex = Union[int, slice, List[int], ndarray, Tensor]

IndexType = str  # 'pos', 'key' or 'all'
ConcatToggle = bool
ActivationKeyConfig = Dict[str, Union[ActivationName, IndexType, ConcatToggle]]

ActivationKey = Union[ActivationIndex, Tuple[ActivationIndex, ActivationKeyConfig]]
