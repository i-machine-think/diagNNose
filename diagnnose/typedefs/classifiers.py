from typing import Dict, Tuple

from torch import Tensor

DataDict = Dict[str, Tensor]

LinearDecoder = Tuple[Tensor, Tensor]
