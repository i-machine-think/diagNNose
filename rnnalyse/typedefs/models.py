from typing import BinaryIO, Dict, List, Tuple, Union

import numpy as np
from torch import Tensor

ActivationName = Tuple[int, str]
ActivationNames = List[ActivationName]

ActivationFiles = Dict[ActivationName, BinaryIO]

ActivationLayer = Dict[str, Tensor]

# Nested dict with embeddings for each activation
FullActivationDict = Dict[int, ActivationLayer]

# Dict with arbitrary number of activations
PartialActivationDict = Dict[ActivationName, Tensor]
PartialArrayDict = Dict[ActivationName, Union[np.ndarray, List[np.ndarray]]]


ParameterDict = FullActivationDict
