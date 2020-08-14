import os
from itertools import product
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from diagnnose.activations.selection_funcs import final_token
from diagnnose.corpus import Corpus, import_corpus
from diagnnose.extract import Extractor
from diagnnose.models import LanguageModel
from diagnnose.tokenizer import Tokenizer
from diagnnose.typedefs import config as config
from diagnnose.typedefs.activations import (
    ActivationDict,
    ActivationName,
    ActivationNames,
)
from diagnnose.utils import __file__ as diagnnose_utils_init
from diagnnose.utils.misc import suppress_print
from diagnnose.utils.pickle import load_pickle

# (layer, name) -> size
SizeDict = Dict[ActivationName, int]


class RecurrentLM(LanguageModel):
    """ Abstract class for LM with intermediate activations """

    device: str = "cpu"
    forget_offset: int = 0
    ih_concat_order: List[str] = ["h", "i"]
    sizes: SizeDict = {}
    split_order: List[str]
    use_char_embs: bool = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.init_states: ActivationDict = {}

    @property
    def num_layers(self) -> int:
        return max(layer for layer, _name in self.sizes) + 1

    @property
    def top_layer(self) -> int:
        return self.num_layers - 1

    @property
    def output_size(self) -> int:
        return self.sizes[self.top_layer, "hx"]

    def init_hidden(self, batch_size: int) -> ActivationDict:
        """Creates a batch of initial states.

        Parameters
        ----------
        batch_size : int
            Size of batch for which states are created.

        Returns
        -------
        init_states : ActivationTensors
            Dictionary mapping hidden and cell state to init tensors.
        """
        init_states = self._expand_batch_size(self.init_states, batch_size)

        for k, v in init_states.items():
            init_states[k] = v.to(self.device)

        return init_states

    def final_hidden(self, hidden: ActivationDict) -> Tensor:
        """ Returns the final hidden state.

        Parameters
        ----------
        hidden : ActivationTensors
            Dictionary of extracted activations.

        Returns
        -------
        final_hidden : Tensor
            Tensor of the final hidden state.
        """
        return hidden[self.top_layer, "hx"].squeeze()

    def nhid(self, activation_name: ActivationName) -> int:
        """ Returns number of hidden units for a (layer, name) tuple.

        If `name` != emb/hx/cx returns the size of (layer, `cx`).
        """
        layer, name = activation_name

        return self.sizes.get((layer, name), self.sizes[layer, "cx"])

    def set_init_states(
        self,
        pickle_path: Optional[str] = None,
        corpus_path: Optional[str] = None,
        use_default: bool = False,
        save_init_states_to: Optional[str] = None,
        tokenizer: Optional[Tokenizer] = None,
    ) -> None:
        """ Set up the initial LM states.

        If no path is provided 0-valued embeddings will be used.
        Note that the loaded init should provide tensors for `hx`
        and `cx` in all layers of the LM.

        Note that `init_states_pickle` takes precedence over
        `init_states_corpus` in case both are provided.

        Parameters
        ----------
        pickle_path : str, optional
            Path to pickled file with initial lstm states. If not
            provided zero-valued init states will be created.
        corpus_path : str, optional
            Path to corpus of which the final hidden state will be used
            as initial states.
        use_default : bool
            Toggle to use the default initial sentence `. <eos>`.
        save_init_states_to : str, optional
            Path to which the newly computed init_states will be saved.
            If not provided these states won't be dumped.
        tokenizer : Tokenizer, optional
            Tokenizer that must be provided when creating the init
            states from a corpus.

        Returns
        -------
        init_states : ActivationTensors
            ActivationTensors containing the init states for each layer.
        """
        if use_default:
            diagnnose_utils_dir = os.path.dirname(diagnnose_utils_init)
            corpus_path = os.path.join(diagnnose_utils_dir, "init_sentence.txt")

        if pickle_path is not None:
            print("Loading extracted init states from file")
            init_states: ActivationDict = load_pickle(pickle_path)
            self._validate(init_states)
        elif corpus_path is not None:
            assert (
                tokenizer is not None
            ), "Tokenizer must be provided when creating init states from corpus"
            print("Creating init states from provided corpus")
            init_states = self._create_init_states_from_corpus(
                corpus_path, tokenizer, save_init_states_to
            )
        else:
            init_states = self.create_zero_states()

        self.init_states = init_states

    def create_zero_states(self, batch_size: int = 1) -> ActivationDict:
        """Zero-initialized states if no init state is provided.

        Parameters
        ----------
        batch_size : int, optional
            Batch size should be provided if it's larger than 1.

        Returns
        -------
        init_states : ActivationTensors
            Dictionary mapping (layer, name) tuple to zero-tensor.
        """
        init_states: ActivationDict = {
            a_name: self.create_zero_state(batch_size, a_name)
            for a_name in product(range(self.num_layers), ["cx", "hx"])
        }

        return init_states

    def create_zero_state(
        self, batch_size: int, activation_name: ActivationName
    ) -> Tensor:
        """ Create single zero tensor for given layer/cell_type.

        Parameters
        ----------
        batch_size : int
            Batch size for model task.
        activation_name : ActivationName
            (layer, name) tuple.

        Returns
        -------
        tensor : Tensor
            Zero-valued tensor of the correct size.
        """
        return torch.zeros((batch_size, self.nhid(activation_name)), dtype=config.DTYPE)

    @suppress_print
    def _create_init_states_from_corpus(
        self,
        init_states_corpus: str,
        tokenizer: Tokenizer,
        save_init_states_to: Optional[str],
    ) -> ActivationDict:
        corpus: Corpus = Corpus.create(init_states_corpus, tokenizer=tokenizer)

        activation_names: ActivationNames = [
            (layer, name) for layer in range(self.num_layers) for name in ["hx", "cx"]
        ]

        self.init_states = self.create_zero_states()
        extractor = Extractor(
            self,
            corpus,
            activation_names,
            activations_dir=save_init_states_to,
            selection_func=final_token,
        )
        init_states = extractor.extract().activation_dict

        return init_states

    def _validate(self, init_states: ActivationDict) -> None:
        """ Performs a simple validation of the new initial states.

        Parameters
        ----------
        init_states: ActivationTensors
            New initial states that should have a structure that
            complies with the dimensions of the language model.
        """
        num_init_layers = max(layer for layer, _name in init_states)
        assert (
            num_init_layers == self.num_layers
        ), "Number of initial layers not correct"

        for (layer, name), size in self.sizes.items():
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

    def _expand_batch_size(
        self, init_states: ActivationDict, batch_size: int
    ) -> ActivationDict:
        """Expands the init_states in the batch dimension."""
        batch_init_states: ActivationDict = {}

        for layer in range(self.num_layers):
            for hc in ["hx", "cx"]:
                # Shape: (batch_size, nhid)
                batch_init_states[layer, hc] = init_states[layer, hc].repeat(
                    batch_size, 1
                )

        return batch_init_states
