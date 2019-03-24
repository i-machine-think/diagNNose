from typing import Any, Dict, Tuple

import numpy as np

from .activations import ActivationName

DataDict = Dict[str, np.ndarray]
ResultsDict = Dict[ActivationName, Dict[str, Any]]

LinearDecoder = Tuple[np.ndarray, np.ndarray]
