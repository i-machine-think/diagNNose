import pickle
import random

import torch


def create_and_dump_dummy_activations(
    num_sentences: int,
    activations_dim: int,
    max_sen_len: int,
    num_classes: int,
    activations_dir: str,
    activations_name: str,
) -> int:
    """ Create and dump activations for a dummy corpus.

    Parameters
    ----------
    num_sentences : int
        Number of sentences in the corpus.
    activations_dim : int
        Dimension of the dummy activations.
    max_sen_len : int
        Maximum sentence length.
    num_classes : int
        Number of label classes.
    activations_dir : str
        Directory to save the activations and corpus to.
    activations_name : str
        Type of activation for which activations will be extracted.

    Returns
    -------
    num_labels : int
        Total number of labels/activations that have been created.
    """

    with open(f"{activations_dir}/{activations_name}.pickle", "wb") as f:
        num_labels = 0
        # Identify activations globally by adding a number on one end
        activation_identifier = 0
        sen_lens = []

        corpus = []

        for i in range(num_sentences):
            # Determine the number of tokens in this sentence
            sen_len = random.randint(1, max_sen_len)
            sen_lens.append((num_labels, num_labels + sen_len))
            num_labels += sen_len
            activations = create_sentence_dummy_activations(
                sen_len, activations_dim, activation_identifier
            )
            activation_identifier += sen_len  # Increment global activation id

            sen = " ".join(
                [
                    str(w)
                    for w in torch.multinomial(
                        torch.arange(num_classes).float(), sen_len, replacement=True
                    )
                ]
            )
            labels = " ".join(
                [
                    str(w)
                    for w in torch.multinomial(
                        torch.arange(num_classes).float(), sen_len, replacement=True
                    )
                ]
            )
            corpus.append("\t".join((sen, labels)))

            pickle.dump(activations, f)

    # Create a new corpus .tsv file
    with open(f"{activations_dir}/corpus.tsv", "w") as f:
        f.write("\n".join(corpus))

    # Create a new activation ranges file, indexed by non-consecutive keys
    with open(f"{activations_dir}/ranges.pickle", "wb") as f:
        ranges = {}
        for i in range(num_sentences):
            ranges[(i + 1) ** 2] = sen_lens[i]
        pickle.dump(ranges, f)

    return num_labels


def create_sentence_dummy_activations(
    sentence_len: int, activations_dim: int, identifier_value_start: int = 0
) -> torch.Tensor:
    """
    Create dummy activations for a single sentence.
    """
    activations = torch.ones(sentence_len, activations_dim)

    # First activation is a vector of ones, second activation a vector of twos and so on
    # (except for an extra identifier dimension which will come in handy later)
    for n in range(sentence_len - 1):
        activations[n + 1 :, :] += 1

    # Add identifier value for each activation
    identifier_values = torch.arange(
        identifier_value_start, identifier_value_start + sentence_len
    )
    activations[:, -1] = identifier_values  # Add values on last dimension of activation

    return activations
