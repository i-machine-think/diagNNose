from typing import BinaryIO, Dict, List, Union, Tuple

import numpy as np
from torch import Tensor


# TODO: Overhaul these names, as they have become cluttered/ambiguous

# (layer, name)
ActivationName = Tuple[int, str]
ActivationNames = List[ActivationName]

ActivationFiles = Dict[ActivationName, BinaryIO]

NamedArrayDict = Dict[str, Union[Tensor, np.ndarray]]

# Nested dict with embeddings for each activation
FullActivationDict = Dict[int, NamedArrayDict]

# Dict with arbitrary number of activations
PartialActivationDict = Dict[ActivationName, Tensor]
PartialArrayDict = Dict[ActivationName, Union[np.ndarray, List[np.ndarray]]]
BatchArrayDict = Dict[int, PartialArrayDict]

ParameterDict = Dict[int, Union[Tensor, np.ndarray]]


# Activation indexing, as done in ActivationReader
ActivationIndex = Union[int, slice, List[int], np.ndarray]

IndexType = str  # 'pos', 'key' or 'all'
ConcatToggle = bool
ActivationKeyConfig = Dict[str, Union[ActivationName, IndexType, ConcatToggle]]

ActivationKey = Union[
    ActivationIndex,
    Tuple[ActivationIndex, ActivationKeyConfig]
]
