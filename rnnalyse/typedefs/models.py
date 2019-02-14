from typing import BinaryIO, Dict, List, Tuple

from torch import Tensor

ActivationName = Tuple[int, str]
ActivationNames = List[ActivationName]

ActivationFiles = Dict[ActivationName, BinaryIO]

ActivationLayer = Dict[str, Tensor]

# Nested dict with embeddings for each activation
FullActivationDict = Dict[int, ActivationLayer]

# Dict with arbitrary number of activations
PartialActivationDict = Dict[ActivationName, Tensor]

ParameterDict = FullActivationDict
