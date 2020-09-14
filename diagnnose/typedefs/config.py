from typing import Any, Dict

import numpy as np
import torch

# group -> arg_name -> value
ConfigDict = Dict[str, Dict[str, Any]]

# group -> arg_name -> arg_attr -> value
# Where arg_attr one of 'help', 'nargs', or 'type'.
ArgDescriptions = Dict[str, Dict[str, Dict[str, Any]]]

# Tensor dtype that will be used in the library. Can be set here to change.
DTYPE = torch.float32
DTYPE_np = np.float32
