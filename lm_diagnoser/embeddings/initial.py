import pickle
from pathlib import Path

import torch

from typedefs.models import FullActivationDict
from models.language_model import LanguageModel


class InitEmbs:
    """Initial embeddings that are passed to LM at start of sequence.

    Attributes
    ----------
    num_layers : int
        Number of layers of the language model
    model : LanguageModel
        LanguageModel containing number of layers and hidden units used.
    activations : FullActivationDict
        Dictionary mapping each activation name to an initial embedding.
    """
    def __init__(self,
                 init_embs_path: str,
                 model: LanguageModel) -> None:
        self.num_layers = model.num_layers
        self.hidden_size = model.hidden_size
        self.activations = self.create_init_embs(init_embs_path)

    def create_init_embs(self, init_embs_path: str) -> FullActivationDict:
        """ Set up the initial LM embeddings.

        If no path is provided 0-initialized embeddings will be used.
        Note that the loaded init should provide tensors for `hx`
        and `cx` in all layers of the LM.

        Parameters
        ----------
        init_embs_path : str
            Path to init embeddings.

        Returns
        -------
        init : FullActivationDict
            FullActivationDict containing init embeddings for each layer.
        """
        if init_embs_path:
            assert Path(init_embs_path).is_file(), 'File does not exist'

            with open(init_embs_path, 'rb') as f:
                init_embs = pickle.load(f)

            self.validate_init_embs(init_embs)

            return init_embs

        return self.create_zero_init_embs()

    def validate_init_embs(self, init_embs: FullActivationDict) -> None:
        """ Performs a simple validation of the new embeddings.

        Parameters
        ----------
        init_embs: FullActivationDict
            Opened initial embeddings that should have a structure that
            complies with the dimensions of the language model.
        """
        assert len(init_embs) == self.num_layers, \
            'Number of initial layers not correct'
        assert all(
            'hx' in a.keys() and 'cx' in a.keys()
            for a in init_embs.values()
        ), 'Initial layer names not correct, should be hx and cx'
        assert len(init_embs[0]['hx']) == self.hidden_size, \
            'Initial activation size is incorrect'

    def create_zero_init_embs(self) -> FullActivationDict:
        """Zero-initialized embeddings if no init embedding has been provided"""
        return {
            l: {
                'hx': torch.zeros(self.hidden_size),
                'cx': torch.zeros(self.hidden_size)
            } for l in range(self.num_layers)
        }
