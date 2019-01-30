from typing import Any, Dict

import numpy as np

from .models import ActivationName

DataDict = Dict[str, np.ndarray]
ResultsDict = Dict[ActivationName, Dict[str, Any]]
