from typing import Callable, Dict, Tuple, Union

from torch import Tensor
from torchtext.data import Example

DataDict = Dict[str, Tensor]

LinearDecoder = Tuple[Tensor, Tensor]

# https://www.aclweb.org/anthology/D19-1275/
# sen_id, w position, batch item -> label
ControlTask = Callable[[int, int, Example], Union[str, int]]
