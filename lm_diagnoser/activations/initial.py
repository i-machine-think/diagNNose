import pickle
from pathlib import Path

import torch

from ..typedefs.models import FullActivationDict
from ..models.language_model import LanguageModel


class InitStates:
    """Initial lstm states that are passed to LM at start of sequence.

    Attributes
    ----------
    num_layers : int
        Number of layers of the language model
    model : LanguageModel
        LanguageModel containing number of layers and hidden units used.
    states : FullActivationDict
        Dictionary mapping each activation name to an initial state.
    """
    def __init__(self,
                 init_lstm_states_path: str,
                 model: LanguageModel) -> None:
        self.num_layers = model.num_layers
        self.hidden_size = model.hidden_size
        self.states = self.create_init_states(init_lstm_states_path)

    def create_init_states(self, init_lstm_states_path: str) -> FullActivationDict:
        """ Set up the initial LM states.

        If no path is provided 0-initialized embeddings will be used.
        Note that the loaded init should provide tensors for `hx`
        and `cx` in all layers of the LM.

        Parameters
        ----------
        init_lstm_states_path : str
            Path to init embeddings.

        Returns
        -------
        init : FullActivationDict
            FullActivationDict containing init embeddings for each layer.
        """
        if init_lstm_states_path:
            assert Path(init_lstm_states_path).is_file(), 'File does not exist'

            with open(init_lstm_states_path, 'rb') as f:
                init_states = pickle.load(f)

            self.validate_init_states(init_states)

            return init_states

        return self.create_zero_init_states()

    def validate_init_states(self, init_states: FullActivationDict) -> None:
        """ Performs a simple validation of the new initial states.

        Parameters
        ----------
        init_states: FullActivationDict
            New initial states that should have a structure that
            complies with the dimensions of the language model.
        """
        assert len(init_states) == self.num_layers, \
            'Number of initial layers not correct'
        assert all(
            'hx' in a.keys() and 'cx' in a.keys()
            for a in init_states.values()
        ), 'Initial layer names not correct, should be hx and cx'
        assert len(init_states[0]['hx']) == self.hidden_size, \
            'Initial activation size is incorrect'

    def create_zero_init_states(self) -> FullActivationDict:
        """Zero-initialized states if no init state has been provided"""
        return {
            l: {
                'hx': torch.zeros(self.hidden_size),
                'cx': torch.zeros(self.hidden_size)
            } for l in range(self.num_layers)
        }
