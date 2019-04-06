from typing import Any

import numpy as np

from rnnalyse.models.language_model import LanguageModel
from rnnalyse.typedefs.activations import ActivationName, DecomposeArrayDict, PartialArrayDict
from rnnalyse.typedefs.classifiers import LinearDecoder


class BaseDecomposer:
    """ Base decomposer which decomposition classes should inherit

    Parameters
    ----------
    decoder : (np.ndarray, np.ndarray) ((num_classes, hidden_dim), (hidden_dim,))
        (Coefficients, bias) tuple of the (linear) decoding layer
    activation_dict : PartialArrayDict
        Dictionary containing the necessary activations for decomposition
    final_index : np.ndarray
        1-d numpy array with index of final element of a batch element.
        Due to masking for sentences of uneven length the final index
        can differ between batch elements.
    layer : int
        The index of the layer on which the decomposition will be done.
    """
    def __init__(self,
                 model: LanguageModel,
                 decoder: LinearDecoder,
                 activation_dict: PartialArrayDict,
                 final_index: np.ndarray,
                 layer: int) -> None:
        self.model = model
        self.decoder_w, self.decoder_b = decoder
        self.activation_dict = activation_dict

        self.final_index = final_index
        self.batch_size = len(final_index)
        self.layer = layer

        self._validate_activation_shapes()
        self._append_init_cell_states()

    def _decompose(self, *arg: Any) -> DecomposeArrayDict:
        raise NotImplementedError

    def decompose(self, *arg: Any, append_bias: bool = False) -> DecomposeArrayDict:
        decomposition = self._decompose(*arg)

        if append_bias:
            bias = self.decompose_bias()
            bias = np.broadcast_to(bias, (self.batch_size, 1, len(bias)))
            for key, arr in decomposition.items():
                decomposition[key] = np.concatenate((arr, bias), axis=1)

        return decomposition

    def decompose_bias(self) -> np.ndarray:
        return np.exp(self.decoder_b)

    def calc_original_logits(self, normalize: bool = False) -> np.ndarray:
        bias = self.decoder_b

        assert (self.layer, 'hx') in self.activation_dict, \
            '\'hx\' should be provided to calculate the original logit'
        final_hidden_state = self.get_final_activations((self.layer, 'hx'))

        original_logit = np.exp(np.ma.dot(final_hidden_state, self.decoder_w.T) + bias)

        if normalize:
            original_logit = (original_logit.T / np.sum(original_logit, axis=1)).T

        return original_logit

    def get_final_activations(self, a_name: ActivationName, offset: int = 0) -> np.ndarray:
        return self.activation_dict[a_name][range(self.batch_size), self.final_index+offset]

    def _validate_activation_shapes(self) -> None:
        pass

    def _append_init_cell_states(self) -> None:
        for layer, name in self.activation_dict:
            if name == 'icx':
                if (layer, 'cx') in self.activation_dict:
                    self.activation_dict[(layer, 'cx')] = np.ma.concatenate((
                        self.activation_dict[(layer, 'icx')],
                        self.activation_dict[(layer, 'cx')]
                    ), axis=1)
                    if (layer, '0cx') in self.activation_dict:
                        self.activation_dict[(layer, 'cx')] = np.ma.concatenate((
                            self.activation_dict[(layer, '0cx')],
                            self.activation_dict[(layer, 'cx')]
                        ), axis=1)
