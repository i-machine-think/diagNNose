from typing import Any, Dict, Set

from torch import float32

ArgDict = Dict[str, Any]
ConfigDict = Dict[str, ArgDict]

RequiredArgs = Set[str]

ArgDescriptions = Dict[str, Dict[str, Dict[str, Any]]]

# Tensor dtype that will be used in the library. Can be set here to change.
DTYPE = float32
