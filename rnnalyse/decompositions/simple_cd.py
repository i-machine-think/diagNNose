import numpy as np


class SimpleCD:
    """
    Implementation of the LSTM decomposition method described in:

    Murdoch and Szlam, Automatic Rule Extraction from Long Short Term
    Memory Networks (2017) https://arxiv.org/pdf/1702.02540.pdf

    Parameters
    ----------
    decoder : np.ndarray (num_classes, hidden_dim)
        Coefficients of the (linear) decoding layer
    output_gate : np.ndarray (num_tokens, hidden_dim)
        Output gate activations for each token
    init_cell_state : np.ndarray (hidden_dim,)
        Initial cell state at start of sentence
    cell_states : np.ndarray (num_tokens, hidden_dim)
        Cell state activations for each token
    """
    def __init__(self,
                 decoder: np.ndarray,
                 output_gate: np.ndarray,
                 cell_states: np.ndarray,
                 init_cell_state: np.ndarray) -> None:
        self.decoder = decoder
        self.output_gates = output_gate
        self.cell_states = cell_states
        self.init_cell_state = np.expand_dims(init_cell_state, 0)

        self.validate_shapes()

    def calc_beta(self) -> np.ndarray:
        prev_cells = np.concatenate((self.init_cell_state, self.cell_states[:-1]))
        cell_diff = np.tanh(self.cell_states) - np.tanh(prev_cells)
        beta = np.exp(np.dot(self.output_gates * cell_diff, self.decoder.T))

        return beta

    def validate_shapes(self) -> None:
        decoder_shape = self.decoder.shape
        output_gate_shape = self.output_gates.shape
        init_cell_shape = self.init_cell_state.shape
        cell_states_shape = self.cell_states.shape

        assert decoder_shape[1] == output_gate_shape[1], \
            f'{decoder_shape[1]} != {output_gate_shape[1]}'
        assert output_gate_shape == cell_states_shape, \
            f'{output_gate_shape} != {cell_states_shape}'
        assert init_cell_shape[1] == cell_states_shape[1], \
            f'{init_cell_shape[1]} != {cell_states_shape[1]}'
