from typing import Any

import numpy as np

from diagnnose.models.language_model import LanguageModel
from diagnnose.typedefs.activations import ActivationName, NamedArrayDict, PartialArrayDict
from diagnnose.typedefs.classifiers import LinearDecoder


class BaseDecomposer:
    """ Base decomposer which decomposition classes should inherit

    Parameters
    ----------
    model : LanguageModel
        LanguageModel for which decomposition will be performed
    activation_dict : PartialArrayDict
        Dictionary containing the necessary activations for decomposition
    decoder : (np.ndarray, np.ndarray) ((num_classes, hidden_dim), (hidden_dim,))
        (Coefficients, bias) tuple of the (linear) decoding layer
    final_index : np.ndarray
        1-d numpy array with index of final element of a batch element.
        Due to masking for sentences of uneven length the final index
        can differ between batch elements.
    """
    def __init__(self,
                 model: LanguageModel,
                 activation_dict: PartialArrayDict,
                 decoder: LinearDecoder,
                 final_index: np.ndarray) -> None:
        self.model = model
        self.decoder_w, self.decoder_b = decoder
        self.activation_dict = activation_dict

        self.final_index = final_index
        self.batch_size = len(final_index)
        self.toplayer = model.num_layers-1

        self._validate_activation_shapes()
        self._append_init_states()

    def _decompose(self, *arg: Any, **kwargs: Any) -> NamedArrayDict:
        raise NotImplementedError

    def decompose(self, *arg: Any, append_bias: bool = False, **kwargs: Any) -> NamedArrayDict:
        decomposition = self._decompose(*arg, **kwargs)

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

        assert (self.toplayer, 'hx') in self.activation_dict, \
            '\'hx\' should be provided to calculate the original logit'
        final_hidden_state = self.get_final_activations((self.toplayer, 'hx'))

        original_logit = np.exp(np.ma.dot(final_hidden_state, self.decoder_w.T) + bias)

        if normalize:
            original_logit = (np.exp(original_logit).T / np.sum(np.exp(original_logit), axis=1)).T

        return original_logit

    def get_final_activations(self, a_name: ActivationName, offset: int = 0) -> np.ndarray:
        return self.activation_dict[a_name][range(self.batch_size), self.final_index+offset]

    def _validate_activation_shapes(self) -> None:
        pass

    def _append_init_states(self) -> None:
        for layer, name in self.activation_dict:
            if name.startswith('i') and name[1:] in ['cx', 'hx']:
                cell_type = name[1:]
                if (layer, cell_type) in self.activation_dict:
                    self.activation_dict[(layer, cell_type)] = np.ma.concatenate((
                        self.activation_dict[(layer, name)],
                        self.activation_dict[(layer, cell_type)]
                    ), axis=1)

                    if cell_type == 'hx' and layer == self.toplayer:
                        self.final_index += 1

                    # 0cx activations should be concatenated in front of the icx activations.
                    if (layer, f'0{cell_type}') in self.activation_dict:
                        self.activation_dict[(layer, cell_type)] = np.ma.concatenate((
                            self.activation_dict[(layer, f'0{cell_type}')],
                            self.activation_dict[(layer, cell_type)]
                        ), axis=1)

                        if cell_type == 'hx' and layer == self.toplayer:
                            self.final_index += 1
