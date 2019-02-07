"""
Useful function for testing.
"""

import pickle
import random

import torch


def create_dummy_activations(num_sentences: int, activations_dim: int, max_tokens: int, num_classes: int,
                             activations_dir: str, activations_name: str):
    with open(f"{activations_dir}/{activations_name}.pickle", "wb") as f:
        num_labels = 0
        activation_identifier = 0  # Identify activations globally by adding on number on one end

        for i in range(num_sentences):
            num_activations = random.randint(1, max_tokens)  # Determine the number of tokens in this sentence
            num_labels += num_activations
            activations = torch.ones(num_activations, activations_dim)

            # First activation is a vector of ones, second activation a vector of twos and so on
            # (except for an extra identifier dimension which will come in handy later)
            for n in range(num_activations-1):
                activations[n+1:, :] += 1

            # Add identifier value for each activation
            identifier_values = torch.arange(activation_identifier, activation_identifier + num_activations)
            activations[:, -1] = identifier_values  # Add values on last dimension of activation
            activation_identifier += num_activations  # Increment global activation id

            pickle.dump(activations, f)

    # Generate some random labels and dump them
    with open(f"{activations_dir}/labels.pickle", "wb") as f:
        labels = torch.rand(num_labels)
        labels = (labels * 10).int() % num_classes
        pickle.dump(labels, f)

    return labels
