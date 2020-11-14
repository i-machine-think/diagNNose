import os
from itertools import product
from typing import Optional

import torch
from transformers import PreTrainedTokenizer

from diagnnose.activations.selection_funcs import final_sen_token
from diagnnose.corpus import Corpus
from diagnnose.extract import Extractor
from diagnnose.typedefs.activations import ActivationDict, ActivationNames
from diagnnose.utils import __file__ as diagnnose_utils_init
from diagnnose.utils.misc import suppress_print
from diagnnose.utils.pickle import load_pickle

from .recurrent_lm import RecurrentLM


def set_init_states(
    model: RecurrentLM,
    pickle_path: Optional[str] = None,
    corpus_path: Optional[str] = None,
    use_default: bool = False,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    save_init_states_to: Optional[str] = None,
) -> None:
    """Set up the initial LM states.

    If no path is provided 0-valued embeddings will be used.
    Note that the loaded init should provide tensors for `hx`
    and `cx` in all layers of the LM.

    Note that `init_states_pickle` takes precedence over
    `init_states_corpus` in case both are provided.

    Parameters
    ----------
    model : RecurrentLM
        A recurrent language model for which the initial states are set.
    pickle_path : str, optional
        Path to pickled file with initial lstm states. If not
        provided zero-valued init states will be created.
    corpus_path : str, optional
        Path to corpus of which the final hidden state will be used
        as initial states.
    use_default : bool
        Toggle to use the default initial sentence `. <eos>`.
    tokenizer : PreTrainedTokenizer, optional
        Tokenizer that must be provided when creating the init
        states from a corpus.
    save_init_states_to : str, optional
        Path to which the newly computed init_states will be saved.
        If not provided these states won't be dumped.

    Returns
    -------
    init_states : ActivationTensors
        ActivationTensors containing the init states for each layer.
    """
    if use_default:
        diagnnose_utils_dir = os.path.dirname(diagnnose_utils_init)
        corpus_path = os.path.join(diagnnose_utils_dir, "init_sentence.txt")

    if pickle_path is not None:
        init_states = _create_init_states_from_pickle(model, pickle_path)
    elif corpus_path is not None:
        init_states = _create_init_states_from_corpus(
            model, corpus_path, tokenizer, save_init_states_to
        )
    else:
        init_states = _create_zero_states(model)

    model.init_states = init_states


def _create_zero_states(model: RecurrentLM) -> ActivationDict:
    """Zero-initialized states if no init state is provided.

    Returns
    -------
    init_states : ActivationTensors
        Dictionary mapping (layer, name) tuple to zero-tensor.
    """
    init_states: ActivationDict = {
        a_name: torch.zeros((1, model.nhid(a_name)))
        for a_name in product(range(model.num_layers), ["cx", "hx"])
    }

    return init_states


@suppress_print
def _create_init_states_from_corpus(
    model: RecurrentLM,
    init_states_corpus: str,
    tokenizer: PreTrainedTokenizer,
    save_init_states_to: Optional[str] = None,
) -> ActivationDict:
    assert (
        tokenizer is not None
    ), "Tokenizer must be provided when creating init states from corpus"

    corpus: Corpus = Corpus.create(init_states_corpus, tokenizer=tokenizer)

    activation_names: ActivationNames = [
        (layer, name) for layer in range(model.num_layers) for name in ["hx", "cx"]
    ]

    extractor = Extractor(
        model,
        corpus,
        activation_names,
        activations_dir=save_init_states_to,
        selection_func=final_sen_token,
    )
    init_states = extractor.extract().activation_dict

    return init_states


def _create_init_states_from_pickle(
    model: RecurrentLM, pickle_path: str
) -> ActivationDict:
    init_states: ActivationDict = load_pickle(pickle_path)

    _validate_init_states_from_pickle(model, init_states)

    return init_states


def _validate_init_states_from_pickle(
    model: RecurrentLM, init_states: ActivationDict
) -> None:
    num_init_layers = max(layer for layer, _name in init_states)
    assert num_init_layers == model.num_layers, "Number of initial layers not correct"

    for (layer, name), size in model.sizes.items():
        if name in ["hx", "cx"]:
            assert (
                layer,
                name,
            ) in init_states.keys(), (
                f"Activation {layer},{name} is not found in init states"
            )

            init_size = init_states[layer, name].size(1)
            assert init_size == size, (
                f"Initial activation size for {name} is incorrect: "
                f"{name}: {init_size}, should be {size}"
            )
