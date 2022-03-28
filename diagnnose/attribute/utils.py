import itertools
from functools import wraps
from math import factorial
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

try:
    # torch 1.5
    from torch._overrides import handle_torch_function, has_torch_function
except ModuleNotFoundError:
    # torch >1.5
    from torch.overrides import handle_torch_function, has_torch_function

MONKEY_PATCH_PERFORMED = False


def monkey_patch():
    """Not all torch functions correctly implement ``__torch_function__``
    yet (i.e. in torch v1.5), as is discussed here:
    https://github.com/pytorch/pytorch/issues/34294

    We override the ``__torch_function__`` behaviour for ``torch.cat``,
    ``torch.stack``, ``Tensor.expand_as``, and ``Tensor.type_as``.
    """
    torch.cat = _monkey_patch_fn(torch.cat)
    torch.stack = _monkey_patch_fn(torch.stack)
    Tensor.expand_as = _monkey_patch_tensor(Tensor.expand_as)
    Tensor.type_as = _monkey_patch_tensor(Tensor.type_as)


def _monkey_patch_fn(original_fn):
    @wraps(original_fn)
    def fn(tensors, dim=0, out=None):
        if not torch.jit.is_scripting():
            if any(type(t) is not Tensor for t in tensors) and has_torch_function(
                tensors
            ):
                return handle_torch_function(fn, tensors, tensors, dim=dim, out=out)
        return original_fn(tensors, dim=dim, out=out)

    return fn


def _monkey_patch_tensor(original_fn):
    @wraps(original_fn)
    def fn(self, other):
        if isinstance(other, Tensor):
            return original_fn(self, other)
        return original_fn(self, other.data)

    return fn


def unwrap(args: Any, attr: str = "data", coalition: Optional[List[int]] = None) -> Any:
    """Unwraps a list of args that might contain ShapleyTensors.

    Can be used to retrieve: 1. The full tensor of each
    ShapleyTensor, 2. The list of contributions, or 3. The sum of
    contributions for a specific coalition.

    Unwrapping is performed recursively. Non-ShapleyTensors are left
    unchanged.

    Parameters
    ----------
    args : Any
        Either the full list of args, or an individual element of that
        list, as unwrapping is performed recursively.
    attr : str, optional
        The ShapleyTensor attribute that should be returned, either
        `data` or `contributions`.
    coalition : List[int], optional
        Optional list of coalition indices. If provided the
        contributions at the indices of the coalition are summed up and
        returned, instead of the full list of contributions.
    """
    if hasattr(args, attr):
        args_attr = getattr(args, attr)
        if coalition is not None:
            return sum_contributions(args_attr, coalition)
        return args_attr
    elif isinstance(args, (Tensor, str)):
        return args
    elif isinstance(args, list):
        return [unwrap(arg, attr, coalition) for arg in args]
    elif isinstance(args, tuple):
        return tuple(unwrap(arg, attr, coalition) for arg in args)

    return args


def sum_contributions(contributions: List[Tensor], coalition: List[int]) -> Tensor:
    """ Sums the contributions that are part of the provided coalition. """
    contributions_sum = sum([contributions[idx] for idx in coalition])
    if isinstance(contributions_sum, int):
        contributions_sum = torch.zeros_like(contributions[0])

    return contributions_sum


def calc_shapley_factors(num_features: int) -> List[Tuple[List[int], int]]:
    """Creates the normalization factors for each subset of features.

    These factors are based on the original Shapley formulation:
    https://en.wikipedia.org/wiki/Shapley_value

    If, for instance, we were to compute these factors for item
    :math:`a` in the set :math:`N = \{a, b, c\}`, we would pass
    :math:`|N|`. This returns the list
    :math:`[([], 2), ([0], 1), ([1], 1), ([0, 1], 2])]`. The first item
    of each tuple should be interpreted as the indices for the set
    :math:`N\setminus\{a\}: (0 \Rightarrow b, 1 \Rightarrow c)`, mapped
    to their factors: :math:`|ids|! \cdot (n - |ids|)!`.

    Parameters
    ----------
    num_features : int
        Number of features for which Shapley values will be computed.

    Returns
    -------
    shapley_factors : List[Tuple[List[int], int]]
        Dictionary mapping a tuple of indices to its corresponding
        normalization factor.
    """
    shapley_factors = []

    for i in range(num_features):
        factor = factorial(i) * factorial(num_features - i - 1)
        for pi in itertools.combinations(range(num_features - 1), i):
            shapley_factors.append((list(pi), factor))

    return shapley_factors


def perm_generator(num_features: int, num_samples: int) -> Iterable[List[int]]:
    """ Generator for feature index permutations. """
    for _ in range(num_samples):
        yield torch.randperm(num_features).tolist()


def init_contributions(
    new_data: Union[Tensor, Sequence[Tensor]], num_features: int
) -> Union[List[Tensor], Sequence[List[Tensor]]]:
    if isinstance(new_data, Sequence):
        return type(new_data)(
            [torch.zeros_like(data) for _ in range(num_features)] for data in new_data
        )
    else:
        return [torch.zeros_like(new_data) for _ in range(num_features)]


def update_contributions(
    contributions: Union[List[Tensor], Sequence[List[Tensor]]],
    f_idx: int,
    output_is_sequential: bool,
    data_with: Union[Tensor, Sequence[Tensor]],
    data_wo: Optional[Union[Tensor, Sequence[Tensor]]] = None,
    factor: float = 1.0,
) -> None:
    if output_is_sequential:
        for output_idx in range(len(data_with)):
            if data_wo is None:
                contributions[output_idx][f_idx] += factor * data_with[output_idx]
            else:
                contributions[output_idx][f_idx] += factor * (
                    data_with[output_idx] - data_wo[output_idx]
                )
    else:
        if data_wo is None:
            contributions[f_idx] += factor * data_with
        else:
            contributions[f_idx] += factor * (data_with - data_wo)


def normalize_contributions(
    contributions: Union[List[Tensor], Sequence[List[Tensor]]],
    factor: int,
    output_is_sequential: bool,
) -> None:
    if output_is_sequential:
        for sub_contributions in contributions:
            for contribution in sub_contributions:
                contribution /= factor
    else:
        for contribution in contributions:
            contribution /= factor


def calc_exact_shapley_values(
    fn: Callable,
    num_features: int,
    shapley_factors: List[Tuple[List[int], int]],
    new_data: Union[Tensor, Sequence[Tensor]],
    baseline_partition: int,
    *args,
    **kwargs,
) -> Union[List[Tensor], Sequence[List[Tensor]]]:
    """Calculates the exact Shapley values for some function fn.

    Note that this procedure grows exponentially in the number of
    features, and should be handled with care.

    Parameters
    ----------
    fn : Callable
        Torch function for which the Shapley values will be computed.
    num_features : int
        Number of features for which contributions will be computed.
    shapley_factors : List[Tuple[List[int], int]]
        List of `Shapley factors` that is computed using
        ``calc_shapley_factors``.
    new_data : Tensor | Sequence[Tensor]
        The output tensor that is currently being decomposed into
        its contributions. We pass this so we can instantiate the
        contribution tensors with correct shape beforehand.
    baseline_partition : int
        Index of the contribution partition to which the baseline fn(0)
        will be added. If we do not add this baseline the contributions
        won't sum up to the full output.
    """
    output_is_sequential: bool = isinstance(new_data, Sequence)
    contributions = init_contributions(new_data, num_features)

    for f_idx in range(num_features):
        other_ids = torch.tensor([i for i in range(num_features) if i != f_idx])

        for coalition_ids, factor in shapley_factors:
            coalition = list(other_ids[coalition_ids])
            args_wo = unwrap(args, attr="contributions", coalition=coalition)

            args_with = unwrap(
                args, attr="contributions", coalition=(coalition + [f_idx])
            )

            data_with = fn(*args_with, **kwargs)
            data_wo = fn(*args_wo, **kwargs)

            update_contributions(
                contributions,
                f_idx,
                output_is_sequential,
                data_with,
                data_wo=data_wo,
                factor=factor,
            )

    normalize_contributions(
        contributions, factorial(num_features), output_is_sequential
    )

    # Add baseline to default feature ([0]).
    zero_input_args = unwrap(args, attr="contributions", coalition=[])
    baseline = fn(*zero_input_args, **kwargs)
    update_contributions(
        contributions, baseline_partition, output_is_sequential, baseline
    )

    return contributions


def calc_sample_shapley_values(
    fn: Callable,
    num_features: int,
    num_samples: int,
    new_data: Union[Tensor, Sequence[Tensor]],
    baseline_partition: int,
    *args,
    **kwargs,
) -> Union[List[Tensor], Sequence[List[Tensor]]]:
    """Calculates the approximate Shapley values for some function fn.

    This procedure is based on that of Castro et al. (2008), and
    approximates Shapley values in polynomial time.

    Parameters
    ----------
    fn : Callable
        Torch function for which the Shapley values will be computed.
    num_features : int
        Number of features for which contributions will be computed.
    num_samples : int
        Number of feature permutation samples. Increasing the number of
        samples will reduce the variance of the approximation.
    new_data : Tensor | Sequence[Tensor]
        The output tensor that is currently being decomposed into
        its contributions. We pass this so we can instantiate the
        contribution tensors with correct shape beforehand.
    baseline_partition : int
        Index of the contribution partition to which the baseline fn(0)
        will be added. If we do not add this baseline the contributions
        won't sum up to the full output.
    """
    contributions = init_contributions(new_data, num_features)
    output_is_sequential: bool = isinstance(new_data, Sequence)

    generator = perm_generator(num_features, num_samples)

    zero_input_args = unwrap(args, attr="contributions", coalition=[])
    baseline = fn(*zero_input_args, **kwargs)

    for sample in generator:
        data_wo = baseline
        for sample_idx, feature_idx in enumerate(sample, start=1):
            coalition = sample[:sample_idx]
            coalition_args = unwrap(args, attr="contributions", coalition=coalition)

            data_with = fn(*coalition_args, **kwargs)

            update_contributions(
                contributions,
                feature_idx,
                output_is_sequential,
                data_with,
                data_wo=data_wo,
            )

            data_wo = data_with

    normalize_contributions(contributions, num_samples, output_is_sequential)

    update_contributions(
        contributions, baseline_partition, output_is_sequential, baseline
    )

    return contributions
