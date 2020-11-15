from collections import namedtuple
from typing import Callable, Union

from torchtext.data import Example

DataDict = namedtuple(
    "DataDict",
    [
        "train_activations",
        "train_labels",
        "train_control_labels",
        "test_activations",
        "test_labels",
        "test_control_labels",
    ],
)

DataSplit = namedtuple("DataSplit", ["activation_reader", "labels", "control_labels"])

DCConfig = namedtuple(
    "DCConfig",
    ["lr", "max_epochs", "rank", "lambda1", "verbose"],
)

# https://www.aclweb.org/anthology/D19-1275/
# w position, batch item -> label
ControlTask = Callable[[int, Example], Union[str, int]]
