from typing import Iterable, Sized

from torch import Tensor

from diagnnose.typedefs.activations import ActivationIndex


def activation_index_to_iterable(activation_index: ActivationIndex) -> Iterable:
    if isinstance(activation_index, Iterable):
        return activation_index

    if isinstance(activation_index, int):
        return [activation_index]

    if isinstance(activation_index, slice):
        assert (
            activation_index.stop is not None
        ), "Stop index of slice should be provided"
        return range(
            activation_index.start or 0,
            activation_index.stop,
            activation_index.step or 1,
        )

    if isinstance(activation_index, Tensor):
        return [int(activation_index[idx]) for idx in range(activation_index.size(0))]

    raise ValueError(
        f"Activation index of incorrect type: {type(activation_index)}, "
        f"should be one of {{int, List[int], np.ndarray or torch.Tensor}}"
    )


def activation_index_len(activation_index: ActivationIndex) -> int:
    activation_iterable = activation_index_to_iterable(activation_index)
    if isinstance(activation_iterable, Sized):
        return len(activation_iterable)

    raise ValueError(
        f"Activation index of incorrect type: {type(activation_index)}, "
        f"should be one of {{int, List[int], np.ndarray or torch.Tensor}}"
    )
