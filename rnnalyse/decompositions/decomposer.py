from typing import Optional

import numpy as np
from sklearn.externals import joblib

from rnnalyse.activations.activation_reader import ActivationReader
from rnnalyse.activations.init_states import InitStates
from rnnalyse.decompositions.simple_cd import SimpleCD
from rnnalyse.typedefs.activations import FullActivationDict
from rnnalyse.utils.paths import trim


class Decomposer:
    def __init__(self,
                 activations_dir: str,
                 decoder_path: str,
                 num_layers: int,
                 hidden_size: int,
                 init_lstm_states_path: Optional[str] = None) -> None:

        self.activation_reader = ActivationReader(activations_dir)
        self.decoder = self._read_decoder(decoder_path)
        self.init_cell_states: FullActivationDict = \
            InitStates(num_layers, hidden_size, init_lstm_states_path).create()

    def decompose(self, sen_id: int, layer: int, decomp_type: str = 'simple') -> np.ndarray:
        output_gates = self.activation_reader[sen_id, 'pos', (1, 'o_g')]
        cell_states = self.activation_reader[sen_id, 'pos', (1, 'cx')]
        init_cell_state = self.init_cell_states[layer]['cx'].numpy()

        decomposer = SimpleCD(self.decoder, output_gates, cell_states, init_cell_state)

        beta = decomposer.calc_beta()
        decomp_probs = beta / beta.sum()

        return decomp_probs

    # TODO: Make this more dynamic by being able to pass the layer index
    @staticmethod
    def _read_decoder(decoder_path: str, decoder_type: str = 'logreg') -> np.ndarray:
        decoder = joblib.load(trim(decoder_path)).coef_

        return decoder
