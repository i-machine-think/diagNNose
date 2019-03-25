from typing import BinaryIO, Dict, List, Union, Tuple

import numpy as np
from torch import Tensor

# (layer, name)
ActivationName = Tuple[int, str]
ActivationNames = List[ActivationName]

ActivationFiles = Dict[ActivationName, BinaryIO]

ActivationLayer = Dict[str, Tensor]

# Nested dict with embeddings for each activation
FullActivationDict = Dict[int, ActivationLayer]

# Dict with arbitrary number of activations
PartialActivationDict = Dict[ActivationName, Tensor]
PartialArrayDict = Dict[ActivationName, Union[np.ndarray, List[np.ndarray]]]

# Decompositions operate per layer on 1 array per type
DecomposeArrayDict = Dict[str, np.ndarray]

ParameterDict = FullActivationDict


# Activation indexing, as done in ActivationReader
ActivationIndex = Union[int, slice, List[int], np.ndarray]

IndexType = str  # 'pos', 'key' or 'all'

ActivationKey = Union[
    ActivationIndex,
    Tuple[ActivationIndex, ActivationName],
    Tuple[ActivationIndex, IndexType],
    Tuple[ActivationIndex, ActivationName, IndexType],
    Tuple[ActivationIndex, IndexType, ActivationName]
]
