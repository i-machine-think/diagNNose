import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
from overrides import overrides
from torch import Tensor, nn

from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.extractors.base_extractor import Extractor
from diagnnose.typedefs.activations import ActivationTensors
from diagnnose.typedefs.corpus import Corpus
from diagnnose.utils.pickle import load_pickle
from diagnnose.utils.misc import suppress_print

SizeDict = Dict[int, Dict[str, int]]


class LanguageModel(ABC, nn.Module):
    """ Abstract class for LM with intermediate activations """

    device: str = "cpu"
    forget_offset: int = 0
    ih_concat_order: List[str] = ["h", "i"]
    sizes: SizeDict = {}
    split_order: List[str]

    def __init__(
        self,
        init_states_pickle: Optional[str] = None,
        init_states_corpus: Optional[str] = None,
        vocab_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.init_states: ActivationTensors = self.set_init_states(
            init_states_pickle=init_states_pickle,
            init_states_corpus=init_states_corpus,
            vocab_path=vocab_path,
        )

    @property
    def num_layers(self) -> int:
        return len(self.sizes)

    @overrides
    @abstractmethod
    def forward(
        self,
        input_: Tensor,
        prev_activations: ActivationTensors,
        compute_out: bool = True,
    ) -> Tuple[Optional[Tensor], ActivationTensors]:
        """ Performs a single forward pass across all rnn layers.

        Parameters
        ----------
        input_ : Tensor
            Tensor containing a batch of token id's at the current
            sentence position.
        prev_activations : TensorDict, optional
            Dict mapping the activation names of the previous hidden
            and cell states to their corresponding Tensors. Defaults to
            None, indicating the initial states will be used.
        compute_out : bool, optional
            Toggles the computation of the final decoder projection.
            If set to False this projection is not calculated.
            Defaults to True.

        Returns
        -------
        out : torch.Tensor, optional
            Torch Tensor of output distribution of vocabulary. If
            `compute_out` is set to True, `out` returns None.
        activations : TensorDict
            Dictionary mapping each layer to each activation name to a
            tensor.
        """

    def init_hidden(self, batch_size: int) -> ActivationTensors:
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
        return self._expand_batch_size(self.init_states, batch_size)

    def set_init_states(
        self,
        init_states_pickle: Optional[str] = None,
        init_states_corpus: Optional[str] = None,
        vocab_path: Optional[str] = None,
    ) -> ActivationTensors:
        """ Set up the initial LM states.

        If no path is provided 0-valued embeddings will be used.
        Note that the loaded init should provide tensors for `hx`
        and `cx` in all layers of the LM.

        Note that `init_states_pickle` takes precedence over
        `init_states_corpus` in case both are provided.

        Arguments
        ---------
        init_states_pickle : int, optional
            Path to pickled file with initial lstm states. If not
            provided zero-valued init states will be created.
        init_states_corpus : int, optional
            Path to corpus of which the final hidden state will be used
            as initial states.
        vocab_path : str, optional
            Path to the model vocabulary, which should a file containing a
            vocab entry at each line. Must be provided when creating
            the init states from a corpus.

        Returns
        -------
        init_states : ActivationTensors
            ActivationTensors containing the init states for each layer.
        """
        if init_states_pickle is not None:
            init_states: ActivationTensors = load_pickle(init_states_pickle)
            self._validate(init_states)
        elif init_states_corpus is not None:
            assert (
                vocab_path is not None
            ), "Vocab path must be provided when creating init states from corpus"
            print("Creating init states from provided corpus")
            init_states = self._create_init_states_from_corpus(init_states_corpus, vocab_path)
        else:
            init_states = self.create_zero_state()

        return init_states

    def create_zero_state(self, batch_size: int = 1) -> ActivationTensors:
        """Zero-initialized states if no init state is provided."""
        init_states: ActivationTensors = {}

        for layer in range(self.num_layers):
            init_states[layer, "cx"] = torch.zeros((batch_size, self.sizes[layer]["c"]))
            init_states[layer, "hx"] = torch.zeros((batch_size, self.sizes[layer]["h"]))

        return init_states

    @suppress_print
    def _create_init_states_from_corpus(
        self, init_states_corpus: str, vocab_path: str
    ) -> ActivationTensors:
        corpus: Corpus = import_corpus(
            init_states_corpus, vocab_path=vocab_path
        )

        self.init_states = self.create_zero_state()
        extractor = Extractor(self, corpus)
        init_states = extractor.extract(
            create_avg_eos=True,
            only_return_avg_eos=True,
        )
        assert init_states is not None

        return init_states

    def _validate(self, init_states: ActivationTensors) -> None:
        """ Performs a simple validation of the new initial states.

        Parameters
        ----------
        init_states: ActivationTensors
            New initial states that should have a structure that
            complies with the dimensions of the language model.
        """
        assert (
            len(init_states) == self.num_layers
        ), "Number of initial layers not correct"

        for layer, layer_size in self.sizes.items():
            for hc in ["h", "c"]:
                assert (
                    layer,
                    f"{hc}x",
                ) in init_states.keys(), (
                    f"Activation {layer},{hc}x is not found in init states"
                )

                init_size = init_states[layer, f"{hc}x"].size(0)
                model_size = self.sizes[layer][hc]
                assert init_size == model_size, (
                    f"Initial activation size for {hc}x is incorrect: "
                    f"{hc}x: {init_size}, should be {model_size}"
                )

    def _expand_batch_size(
        self, init_states: ActivationTensors, batch_size: int
    ) -> ActivationTensors:
        """Expands the init_states in the batch dimension."""
        batch_init_states: ActivationTensors = {}

        for layer in range(self.num_layers):
            for hc in ["hx", "cx"]:
                batch_init_states[layer, hc] = torch.repeat_interleave(
                    init_states[layer, hc], batch_size, dim=0
                )

        return batch_init_states
