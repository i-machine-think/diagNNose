from itertools import combinations
from math import factorial
from typing import Callable, Dict, Sequence
import numpy as np
import torch
from scipy.special import expit as sigmoid
from torch import Tensor

import diagnnose.typedefs.config as config


def calc_shapley_factors(n: int) -> Dict[Sequence[int], int]:
    """Creates the normalization factors for each subset of `n` items.

    These factors are based on the original Shapley formulation:
    https://en.wikipedia.org/wiki/Shapley_value

    If, for instance, we were to compute these factors for item
    :math:`a` in the set :math:`N = \{a, b, c\}`, we would pass
    :math:`|N|-1`. This returns the dict
    :math:`\{(): 2, (0,): 1, (1,): 1, (0, 1): 2\}`, whose keys should
    be interpreted as the indices for the set
    :math:`N\setminus\{a\}: (0 \Rightarrow b, 1 \Rightarrow c)`, mapped
    to their factors: :math:`|ids|! \cdot (n - |ids|)!`.

    Parameters
    ----------
    n : int
        Set size (i.e. number of features) for which Shapley values will
        be computed.

    Returns
    -------
    shapley_factors : Dict[Sequence[int], int]
        Dictionary mapping a tuple of indices to its corresponding
        normalization factor.
    """
    shapley_factors = {}

    for i in range(n + 1):
        amount = factorial(i) * factorial(n - i)
        for pi in combinations(range(n), i):
            shapley_factors[pi] = amount

    return shapley_factors


def calc_full_shapley_values(
    tensor: Tensor, func: Callable[[np.ndarray], np.ndarray], bias_as_baseline: bool = False
) -> Tensor:
    """Computes Shapley values for a summed tensor over tanh/sigmoid.

    In more abstract terms, it's an exact calculation of the following:
    :math:`f(\sum_i(x_i)) = \sum_i(R_i(f))`

    This computation grows exponential in the number of input features,
    and is able to create exact Shapley values up to ~20 features.

    Intermediate summed tensors and function results are stored
    dynamically to save time.

    The resulting tensor of Shapley values satisfies the following:
    func(tensor.sum(dim=1)) == shapley_values.sum(dim=1)

    Parameters
    ----------
    tensor : Tensor
        Tensor of size (batch_size, n_features, nhid).
    func : Callable[[np.ndarray], np.ndarray]
        Either `np.tanh` or `scipy.special.expit` (sigmoid).
    bias_as_baseline : bool, optional
        If set to True the bias will be used as a baseline, instead of
        an contributing feature. The bias is assumed to be situated in
        the input tensor as final entry: tensor[:, -1].

    Returns
    -------
    shapley_values : Tensor
        Tensor of size (batch_size, n_features, nhid) containing the
        computed Shapley values.
    """
    batch_size, n_features, nhid = tensor.shape
    # Shapley values are computed for numpy arrays, as it empirically turned out to be faster.
    tensor = tensor.numpy()

    baseline = np.zeros((batch_size, nhid), dtype=config.DTYPE_np)
    if bias_as_baseline:
        n_features -= 1
        baseline = tensor[:, -1]

    # Compute all subset indices + normalization factors
    shapley_coalitions = calc_shapley_factors(n_features - 1)

    shapley_values = np.zeros((batch_size, n_features, nhid), dtype=config.DTYPE_np)

    for idx in range(n_features):
        # Create index array of indices without `idx`, i.e. N\{idx}.
        sub_idx = np.array([i for i in range(n_features) if i != idx])

        for coalition_idx, factor in shapley_coalitions.items():
            coalition_idx = list(coalition_idx)

            # Indices of features in coalition minus the current feature
            idx_wo = tuple(sub_idx[coalition_idx])

            # Create summation over subset of features
            sum_wo = np.sum(tensor[:, idx_wo, :], axis=1) + baseline
            sum_with = sum_wo + tensor[:, idx]

            # Calculate activation values over summed features
            func_wo = func(sum_wo)
            func_with = func(sum_with)

            func_diff = func_with - func_wo
            func_diff *= factor

            shapley_values[:, idx] += func_diff

    shapley_values /= factorial(n_features)

    if bias_as_baseline:
        baseline = func(np.expand_dims(baseline, 1))
        shapley_values = np.concatenate((shapley_values, baseline), axis=1)

    return torch.from_numpy(shapley_values)
