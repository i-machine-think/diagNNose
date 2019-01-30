"""
Create initial representations which are the average of all final sentence representations in a corpus.
"""

import torch
import pickle

from models.language_model import LanguageModel
from typedefs.corpus import LabeledCorpus
from embeddings.initial import InitEmbs


def create_average_eos_representations(model: LanguageModel,
                                       corpus: LabeledCorpus,
                                       representation_path: str):
    eos_representations = {
        layer: {"hx": [], "cx": []}
        for layer in range(model.num_layers)
    }

    init_embs = InitEmbs("", model)
    activations = init_embs.activations

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
    with open(representation_path, "wb") as f:
        pickle.dump(eos_representations, f)
