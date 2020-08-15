from itertools import combinations
from math import factorial
from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor


def unwrap(args: Any, attr: str = "data", coalition: Optional[List[int]] = None) -> Any:
    """ Unwraps a list of args that might contain ShapleyTensors.

    Can be used to retrieve 1. The full tensor of each
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
        attr = getattr(args, attr)
        if coalition is not None:
            return sum_contributions(attr, coalition)
        return attr
    elif isinstance(args, (Tensor, str)):
        return args
    elif isinstance(args, (list, tuple)):
        iterable_type = type(args)
        return iterable_type(map(unwrap, args))

    return args


def sum_contributions(contributions: List[Tensor], coalition: List[int]) -> Tensor:
    """ Sums the contributions that are part of the provided coalition. """
    contributions_sum = sum([contributions[idx] for idx in coalition])
    if isinstance(contributions_sum, int):
        contributions_sum = torch.zeros_like(contributions[0])

    return contributions_sum


def calc_shapley_factors(n: int) -> List[Tuple[List[int], int]]:
    """Creates the normalization factors for each subset of `n` items.

    These factors are based on the original Shapley formulation:
    https://en.wikipedia.org/wiki/Shapley_value

    If, for instance, we were to compute these factors for item
    :math:`a` in the set :math:`N = \{a, b, c\}`, we would pass
    :math:`|N|-1`. This returns the list
    :math:`[([], 2), ([0], 1), ([1], 1), ([0, 1], 2])`. The first item
    of each tuple should be interpreted as the indices for the set
    :math:`N\setminus\{a\}: (0 \Rightarrow b, 1 \Rightarrow c)`, mapped
    to their factors: :math:`|ids|! \cdot (n - |ids|)!`.

    Parameters
    ----------
    n : int
        Set size (i.e. number of features) for which Shapley values will
        be computed.

    Returns
    -------
    shapley_factors : List[Tuple[List[int], int]]
        Dictionary mapping a tuple of indices to its corresponding
        normalization factor.
    """
    shapley_factors = []

    for i in range(n + 1):
        factor = factorial(i) * factorial(n - i)
        for pi in combinations(range(n), i):
            shapley_factors.append((list(pi), factor))

    return shapley_factors
