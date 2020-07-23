from typing import Iterable, Optional, Sized

from torch import Tensor, long

from diagnnose.typedefs.activations import ActivationIndex


def activation_index_to_iterable(
    activation_index: ActivationIndex, stop_index: Optional[int] = None
) -> Iterable:
    """ Transforms an activation index into an iterable object. """
    if isinstance(activation_index, Tensor):
        activation_index = activation_index.to(long)

    if isinstance(activation_index, Iterable):
        return activation_index

    if isinstance(activation_index, int):
        return [activation_index]

    if isinstance(activation_index, slice):
        stop_index = activation_index.stop or stop_index
        assert stop_index is not None, "Stop index of slice should be provided"
        return range(
            activation_index.start or 0, stop_index, activation_index.step or 1
        )

    raise ValueError(
        f"Activation index of incorrect type: {type(activation_index)}, "
        f"should be one of {{int, List[int], np.ndarray or torch.Tensor}}"
    )


def activation_index_len(activation_index: ActivationIndex) -> int:
    """ Returns the number of items in an activation index. """
    activation_iterable = activation_index_to_iterable(activation_index)
    if isinstance(activation_iterable, Sized):
        return len(activation_iterable)

    raise ValueError(
        f"Activation index of incorrect type: {type(activation_index)}, "
        f"should be one of {{int, List[int], np.ndarray or torch.Tensor}}"
    )
