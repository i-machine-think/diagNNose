from typing import Optional, Union

import numpy as np
import torch

from diagnnose.models.language_model import LanguageModel
from diagnnose.typedefs.activations import FullActivationDict
from diagnnose.utils.paths import load_pickle


class InitStates:
    """Initial lstm states that are passed to LM at start of sequence.

    Attributes
    ----------
    model : LanguageModel
        Language model for which init states will be created.
    init_lstm_states_path : str, optional
        Path to pickled file with initial lstm states. If not provided
        zero-valued init states will be created.
    """

    def __init__(
        self, model: LanguageModel, init_lstm_states_path: Optional[str] = None
    ) -> None:
        self.sizes = model.sizes
        self.num_layers = model.num_layers

        self.init_lstm_states_path = init_lstm_states_path

        self.use_np_arrays = model.array_type == "numpy"

    def create(self, batch_size: int = 1) -> FullActivationDict:
        """ Set up the initial LM states.

        If no path is provided 0-initialized embeddings will be used.
        Note that the loaded init should provide tensors for `hx`
        and `cx` in all layers of the LM.

        Returns
        -------
        init_states : FullActivationDict
            FullActivationDict containing init embeddings for each layer.
        """
        if self.init_lstm_states_path is not None:
            init_states: FullActivationDict = load_pickle(self.init_lstm_states_path)

            self._validate(init_states)

            init_states = self._expand_batch_size(init_states, batch_size)

            return init_states

        return self.create_zero_init_states(batch_size)

    def _validate(self, init_states: FullActivationDict) -> None:
        """ Performs a simple validation of the new initial states.

        Parameters
        ----------
        init_states: FullActivationDict
            New initial states that should have a structure that
            complies with the dimensions of the language model.
        """
        assert (
            len(init_states) == self.num_layers
        ), "Number of initial layers not correct"
        for layer, layer_size in self.sizes.items():
            init_state_dict = init_states[layer]

            assert (
                "hx" in init_state_dict.keys() and "cx" in init_state_dict.keys()
            ), "Initial layer names not correct, should be hx and cx"

            assert len(init_state_dict["hx"]) == self.sizes[layer]["h"], (
                "Initial activation size for hx is incorrect: "
                f'hx: {len(init_state_dict["hx"])}, should be {self.sizes[layer]["h"]}'
            )

            assert len(init_state_dict["cx"]) == self.sizes[layer]["c"], (
                "Initial activation size for cx is incorrect: "
                f'cx: {len(init_state_dict["cx"])}, should be {self.sizes[layer]["c"]}'
            )

    def _expand_batch_size(
        self, init_states: FullActivationDict, batch_size: int
    ) -> FullActivationDict:
        return {
            l: {
                "cx": np.repeat(
                    init_states[l]["cx"][np.newaxis, :], batch_size, axis=0
                ),
                "hx": np.repeat(
                    init_states[l]["hx"][np.newaxis, :], batch_size, axis=0
                ),
            }
            for l in range(self.num_layers)
        }

    def create_zero_init_states(
        self, batch_size: Optional[int] = None
    ) -> FullActivationDict:
        """Zero-initialized states if no init state has been provided"""
        return {
            l: {
                "cx": self._create_zero_state(self.sizes[l]["c"], batch_size),
                "hx": self._create_zero_state(self.sizes[l]["h"], batch_size),
            }
            for l in range(self.num_layers)
        }

    def _create_zero_state(
        self, size: int, batch_size: Optional[int] = None
    ) -> Union[torch.Tensor, np.ndarray]:
        if self.use_np_arrays:
            return np.zeros((batch_size, size), dtype=np.float32)
        return torch.zeros((batch_size, size))
