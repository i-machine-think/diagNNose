from typing import List, Optional, Union

from sklearn.externals import joblib

from rnnalyse.activations.activation_reader import ActivationReader
from rnnalyse.activations.init_states import InitStates
from rnnalyse.decompositions.simple_cd import SimpleCD
from rnnalyse.typedefs.activations import FullActivationDict, DecomposeArrayDict
from rnnalyse.typedefs.classifiers import LinearDecoder
from rnnalyse.utils.paths import trim

from .base_decomposer import BaseDecomposer


class DecomposerFactory:
    def __init__(self,
                 activations_dir: str,
                 decoder: Union[str, LinearDecoder],
                 num_layers: int,
                 hidden_size: int,
                 init_lstm_states_path: Optional[str] = None) -> None:

        self.activation_reader = ActivationReader(activations_dir)

        if isinstance(decoder, str):
            self.decoder_w, self.decoder_b = self._read_decoder(decoder)
        else:
            self.decoder_w, self.decoder_b = decoder

        self.init_cell_state: FullActivationDict = \
            InitStates(num_layers, hidden_size, init_lstm_states_path).create()

        self.zero_cell_state = InitStates(num_layers, hidden_size).create()

    def create(self,
               sen_id: int,
               layer: int,
               index: slice = slice(None, None, None),
               classes: Union[slice, List[int]] = slice(None, None, None)) -> BaseDecomposer:

        activations = self._create_activations(sen_id, layer, index)

        decoder = self.decoder_w[classes], self.decoder_b[classes]

        decomposer = SimpleCD(decoder, activations)

        return decomposer

    def _create_activations(self,
                            sen_id: int,
                            layer: int,
                            index: slice = slice(None, None, None)) -> DecomposeArrayDict:
        forget_gates = self.activation_reader[sen_id, 'key', (layer, 'f_g')][index]
        output_gates = self.activation_reader[sen_id, 'key', (layer, 'o_g')]
        hidden_states = self.activation_reader[sen_id, 'key', (layer, 'hx')]

        final_index = output_gates.shape[0] - 1 if index.stop is None else index.stop - 1
        final_output_gate = output_gates[final_index]
        final_hidden_state = hidden_states[final_index]

        cell_states = self.activation_reader[sen_id, 'key', (layer, 'cx')]
        if index.start == 0 or index.start is None:
            init_cell_state = self.init_cell_state[layer]['cx'].numpy()
        else:
            init_cell_state = cell_states[index.start-1]
        cell_states = cell_states[index]
        zero_cell_state = self.zero_cell_state[layer]['cx'].numpy()

        return {
            'f_g': forget_gates,
            'o_g': final_output_gate,
            'hx': final_hidden_state,
            'cx': cell_states,
            'icx': init_cell_state,
            '0cx': zero_cell_state,
        }

    @staticmethod
    def _read_decoder(decoder_path: str) -> LinearDecoder:
        classifier = joblib.load(trim(decoder_path))
        decoder_w = classifier.coef_
        decoder_b = classifier.intercept_

        return decoder_w, decoder_b
