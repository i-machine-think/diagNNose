"""
Create initial representations which are the average of all final sentence representations in a corpus.
"""

import torch
import pickle

from ..models.language_model import LanguageModel
from ..typedefs.corpus import LabeledCorpus
from ..activations.initial import InitStates


def create_average_eos_states(model: LanguageModel,
                              corpus: LabeledCorpus,
                              save_path: str) -> None:
    """
    Create initial Language Model activations that correspond to the average end-of-sentence
    activations of sentences in a corpus.

    Parameters
    ----------
    model: LanguageModel
        RNN Language Model used to extract activations.
    corpus: LabeledCorpus
        Corpus whose sentences are used to extract activations.
    save_path: str
        Path to save the activations to.
    """
    eos_representations = {
        layer: {"hx": [], "cx": []}
        for layer in range(model.num_layers)
    }

    init_states = InitStates(model)
    activations = init_states.states

    # Extract representations
    for sentence in corpus.values():
        for token in sentence.sen:
            out, activations = model.forward(token, activations)

        for layer in activations:
            for activation_type in ["hx", "cx"]:
                eos_representations[layer][activation_type].append(activations[layer][activation_type])

    # Calculate average
    for layer in activations:
        for activation_type in ["hx", "cx"]:
            avg_eos_representations = torch.stack(eos_representations[layer][activation_type])
            avg_eos_representations = avg_eos_representations.mean(dim=0)
            eos_representations[layer][activation_type] = avg_eos_representations

    # Save to file
    with open(save_path, "wb") as f:
        pickle.dump(eos_representations, f)
