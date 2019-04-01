from typing import List, Optional, Tuple, Union

import numpy as np
from sklearn.externals import joblib

from rnnalyse.activations.activation_reader import ActivationKey, ActivationReader
from rnnalyse.activations.init_states import InitStates
from rnnalyse.decompositions.simple_cd import SimpleCD
from rnnalyse.typedefs.activations import DecomposeArrayDict, FullActivationDict
from rnnalyse.typedefs.classifiers import LinearDecoder
from rnnalyse.utils.paths import trim

from .base_decomposer import BaseDecomposer


class DecomposerFactory:
    """ Creates a BaseDecomposer class for activation decomposition

    Parameters
    ----------
    activations_dir : str
        Path to folder containing extracted activations
    decoder : Union[str, LinearDecoder]
        Path to a pickled decoder or a (w,b) decoder tuple
    num_layers : int
        Number of layers in the language model
    hidden_size : int
        Number of hidden units in the language model
    init_lstm_states_path : str, optional
        Defaults to zero-embeddings, otherwise loads in pickled initial
        cell states.
    """

    def __init__(self,
                 activations_dir: str,
                 decoder: Union[str, LinearDecoder],
                 num_layers: int,
                 hidden_size: int,
                 init_lstm_states_path: Optional[str] = None) -> None:

        self.activation_reader = ActivationReader(activations_dir, store_multiple_activations=True)

        if isinstance(decoder, str):
            self.decoder_w, self.decoder_b = self._read_decoder(decoder)
        else:
            self.decoder_w, self.decoder_b = decoder

        self.hidden_size = hidden_size
        self.init_cell_state: FullActivationDict = \
            InitStates(num_layers, hidden_size, init_lstm_states_path).create()

    # TODO: rename args (sen_index/index is confusing!)
    def create(self,
               sen_index: ActivationKey,
               layer: int,
               index: slice = slice(None, None, None),
               classes: Union[slice, List[int]] = slice(None, None, None)) -> BaseDecomposer:

        activations, batch_size, final_index = self._create_activations(sen_index, layer, index)

        decoder = self.decoder_w[classes], self.decoder_b[classes]

        decomposer = SimpleCD(decoder, activations, batch_size, final_index)

        return decomposer

    # TODO: refactor
    def _create_activations(self,
                            sen_index: ActivationKey,
                            layer: int,
                            index: slice = slice(None, None, None)
                            ) -> Tuple[DecomposeArrayDict, int, np.ndarray]:
        activation_key_config = {'indextype': 'key', 'concat': False, 'a_name': (layer, 'f_g')}
        forget_gates = self.activation_reader[sen_index, activation_key_config][:, index]

        activation_key_config['a_name'] = (layer, 'o_g')
        output_gates = self.activation_reader[sen_index, activation_key_config]

        activation_key_config['a_name'] = (layer, 'hx')
        hidden_states = self.activation_reader[sen_index, activation_key_config]

        batch_size = output_gates.shape[0]

        if index.stop is None:
            # Recomputes the sentences length based on the size of the mask,
            final_index = np.sum(np.all(1 - output_gates.mask, axis=2), axis=1) - 1
        else:
            final_index = np.array([index.stop - 1] * batch_size)
        final_output_gate = output_gates[range(batch_size), [final_index]][0]
        final_hidden_state = hidden_states[range(batch_size), [final_index]][0]

        activation_key_config['a_name'] = (layer, 'cx')
        cell_states = self.activation_reader[sen_index, activation_key_config]

        if index.start == 0 or index.start is None:
            init_cell_state = self.init_cell_state[layer]['cx'].numpy()
            init_cell_state = np.tile(init_cell_state, (batch_size, 1))
        else:
            init_cell_state = cell_states[:, index.start - 1]
        init_cell_state = np.expand_dims(init_cell_state, 1)
        zero_cell_state = np.zeros((batch_size, 1, self.hidden_size))
        cell_states = cell_states[:, index]

        start = index.start if index.start is not None else 0
        final_index -= start

        return {
                   'f_g': forget_gates,
                   'o_g': final_output_gate,
                   'hx': final_hidden_state,
                   'cx': cell_states,
                   'icx': init_cell_state,
                   '0cx': zero_cell_state,
               }, batch_size, final_index

    @staticmethod
    def _read_decoder(decoder_path: str) -> LinearDecoder:
        classifier = joblib.load(trim(decoder_path))
        decoder_w = classifier.coef_
        decoder_b = classifier.intercept_

        return decoder_w, decoder_b
