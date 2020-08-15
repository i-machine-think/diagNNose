from typing import Any, Dict, Set, Union

import numpy as np
import torch

ArgDict = Dict[str, Any]
ConfigDict = Dict[str, Union[ArgDict, str]]

RequiredArgs = Set[str]

ArgDescriptions = Dict[str, Dict[str, Dict[str, Any]]]

# Tensor dtype that will be used in the library. Can be set here to change.
DTYPE = torch.float32
DTYPE_np = np.float32
