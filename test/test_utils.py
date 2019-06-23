"""
Useful function for testing.
"""

from contextlib import redirect_stdout
from functools import wraps
import io
import pickle
import random
from typing import Callable, Any, Type

import torch


def create_and_dump_dummy_activations(num_sentences: int,
                                      activations_dim: int,
                                      max_tokens: int,
                                      num_classes: int,
                                      activations_dir: str,
                                      activations_name: str) -> torch.Tensor:
    """ Create and dump activations for a fictitious corpus. """

    with open(f"{activations_dir}/{activations_name}.pickle", "wb") as f:
        num_labels = 0
        activation_identifier = 0  # Identify activations globally by adding a number on one end
        sen_lens = []

        for i in range(num_sentences):
            # Determine the number of tokens in this sentence
            num_activations = random.randint(1, max_tokens)
            sen_lens.append((num_labels, num_labels + num_activations))
            num_labels += num_activations
            activations = create_sentence_dummy_activations(
                num_activations, activations_dim, activation_identifier
            )
            activation_identifier += num_activations  # Increment global activation id

            pickle.dump(activations, f)

    # Generate some random labels and dump them
    with open(f"{activations_dir}/labels.pickle", "wb") as f:
        labels = torch.rand(num_labels)
        labels = (labels * 10).int() % num_classes
        pickle.dump(labels, f)

    with open(f"{activations_dir}/ranges.pickle", "wb") as f:
        ranges = {}
        for i in range(num_sentences):
            ranges[(i+1)**2] = sen_lens[i]
        pickle.dump(ranges, f)

    return labels


def create_sentence_dummy_activations(sentence_len: int, activations_dim: int,
                                      identifier_value_start: int = 0) -> torch.Tensor:
    """
    Create dummy activations for a single sentence.
    """
    activations = torch.ones(sentence_len, activations_dim)

    # First activation is a vector of ones, second activation a vector of twos and so on
    # (except for an extra identifier dimension which will come in handy later)
    for n in range(sentence_len - 1):
        activations[n + 1:, :] += 1

    # Add identifier value for each activation
    identifier_values = torch.arange(identifier_value_start, identifier_value_start + sentence_len)
    activations[:, -1] = identifier_values  # Add values on last dimension of activation

    return activations


def suppress_print(func: Callable) -> Callable:
    """
    Function decorator to suppress output via print for testing purposed. Thanks to
    https://codingdose.info/2018/03/22/supress-print-output-in-python/ for text "entrapment".
    """
    @wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> Any:

        trap = io.StringIO()

        with redirect_stdout(trap):
            result = func(*args, **kwargs)

        return result

    return wrapped

