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

    These factors are based on the originaly Shapley formulation:
    https://en.wikipedia.org/wiki/Shapley_value

    If, for instance, we were to compute these factors for item `a` in
    the set N = {a, b, c}, we would pass `|N|-1`. This returns:
      `{(): 2, (0,): 1, (1,): 1, (0, 1): 2}`,
    which should be interpreted as the indices for the set N\\{a}:
    (0: b, 1: c), mapped to their factors: len(ids)! * (n - len(ids))!

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

    for i in range(n+1):
        amount = factorial(i) * factorial(n - i)
        for pi in combinations(range(n), i):
            shapley_factors[pi] = amount

    return shapley_factors


def calc_shapley_values(
    tensor: Tensor,
    func: Callable[[Tensor], Tensor],
    use_numpy: bool = False
) -> Tensor:
    """Computes Shapley values for a summed tensor over tanh/sigmoid.

    In more abstract terms, it's an exact calculation of the following:
    f(sum_i(x_i)) = sum_i(S_f(x_i)), f = tanh/sigmoid

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
    func : Callable[[Tensor], Tensor]
        Either `tanh` or `sigmoid`.
    use_numpy : bool, optional
        Because numpy and torch run have their own pros and cons, numpy
        seems to be better suited when `n_features` is large, and torch
        when `batch_size` or `nhid` is large.

    Returns
    -------
    shapley_values : Tensor
        Tensor of size (batch_size, n_features, nhid) containing the
        computed Shapley values.
    """
    batch_size, n_features, nhid = tensor.shape

    # Compute all subset indices + normalization factors
    shapley_coalitions = calc_shapley_factors(n_features-1)

    func_dict = {}
    sum_dict = {}

    if use_numpy:
        shapley_values = np.zeros((batch_size, n_features, nhid))
    else:
        shapley_values = torch.zeros((batch_size, n_features, nhid), dtype=config.DTYPE)

    for idx in range(n_features):
        # Create index array of indices without `idx`, i.e. N\{idx}.
        if use_numpy:
            sub_idx = np.array([i for i in range(n_features) if i != idx])
        else:
            sub_idx = torch.tensor([i for i in range(n_features) if i != idx])

        for coalition_idx, factor in shapley_coalitions.items():
            coalition_idx = list(coalition_idx)

            # orig_idx contains the indices of the full feature set
            orig_idx_wo = tuple(sub_idx[coalition_idx])
            orig_idx_with = orig_idx_wo + (idx,)

            # Create summation over subset of features
            if orig_idx_wo not in sum_dict:
                if use_numpy:
                    sum_dict[orig_idx_wo] = np.sum(tensor[:, orig_idx_wo, :], axis=1)
                else:
                    sum_dict[orig_idx_wo] = torch.sum(tensor[:, orig_idx_wo, :], dim=1)
            if orig_idx_with not in sum_dict:
                sum_dict[orig_idx_with] = sum_dict[orig_idx_wo] + tensor[:, idx]

            # Calculate activation values over summed features
            if orig_idx_wo in func_dict:
                func_wo = func_dict[orig_idx_wo]
            else:
                # Don't subtract baseline values, this redistributes the baseline value of f(0)
                # over all features, allowing the completeness axiom to hold (details in thesis).
                if func in [torch.sigmoid, sigmoid] and len(orig_idx_wo) == 0:
                    if use_numpy:
                        func_wo = np.zeros((batch_size, nhid))
                    else:
                        func_wo = torch.zeros((batch_size, nhid), dtype=config.DTYPE)
                else:
                    func_wo = func(sum_dict[orig_idx_wo])
                func_dict[orig_idx_wo] = func_wo

            if orig_idx_with in func_dict:
                func_with = func_dict[orig_idx_with]
            else:
                func_with = func(sum_dict[orig_idx_with])
                func_dict[orig_idx_with] = func_with

            func_diff = func_with - func_wo
            func_diff *= factor

            shapley_values[:, idx] += func_diff

    shapley_values /= factorial(n_features)

    return shapley_values
