from itertools import combinations
from math import factorial
from typing import Callable, Dict, List, Sequence

import torch
from torch import Tensor


# Activation linearizations as described in chapter 3.2.2
def shapley_three(
    tensors: List[Tensor], gate: Callable[[Tensor], Tensor] = torch.tanh
) -> List[Tensor]:
    a, b, c = tensors
    ac = gate(a + c)
    ab = gate(a + b)
    bc = gate(b + c)
    abc = gate(a + b + c)
    a = gate(a)
    b = gate(b)
    c = gate(c)

    a_contrib = (1 / 6) * (2 * (abc - bc) + (ab - b) + (ac - c) + 2 * a)
    b_contrib = (1 / 6) * (2 * (abc - ac) + (ab - a) + (bc - c) + 2 * b)
    c_contrib = (1 / 6) * (2 * (abc - ab) + (bc - b) + (ac - a) + 2 * c)

    return [a_contrib, b_contrib, c_contrib]


def shapley_two(
    tensors: List[Tensor], gate: Callable[[Tensor], Tensor] = torch.tanh
) -> List[Tensor]:
    a, b = tensors
    ab = gate(a + b)
    a = gate(a)
    b = gate(b)

    a_contrib = 0.5 * (a + (ab - b))
    b_contrib = 0.5 * (b + (ab - a))

    return [a_contrib, b_contrib]


# The original implementation sets the bias term fixed
# to the first position in the Shapley approximation.
def shapley_three_fixed(
    tensors: List[Tensor], gate: Callable[[Tensor], Tensor] = torch.tanh
) -> List[Tensor]:
    a, b, c = tensors
    ac = gate(a + c)
    bc = gate(b + c)
    abc = gate(a + b + c)

    c_contrib = gate(c)

    a_contrib = (1 / 2) * ((abc - bc) + (ac - c_contrib))
    b_contrib = (1 / 2) * ((abc - ac) + (bc - c_contrib))

    return [a_contrib, b_contrib, c_contrib]
