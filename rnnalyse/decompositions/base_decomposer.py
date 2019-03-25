import numpy as np

from rnnalyse.typedefs.activations import DecomposeArrayDict
from rnnalyse.typedefs.classifiers import LinearDecoder


class BaseDecomposer:
    """ Base decomposer which decomposition classes should inherit

    Parameters
    ----------
    decoder : (np.ndarray, np.ndarray) ((num_classes, hidden_dim), (hidden_dim,))
        (Coefficients, bias) tuple of the (linear) decoding layer
    activations : PartialArrayDict
        (layer, name) -> Array dictionary containing:
        'f_g' : np.ndarray (num_tokens, hidden_dim)
            Forget gate activations for each token
        'o_g' : np.ndarray (hidden_dim,)
            Final output gate activation of the sequence
        'hx' : np.ndarray (hidden_dim,)
            Final hidden state of the sequence
        'cx' : np.ndarray (num_tokens, hidden_dim)
            Cell state activations for each token
        'icx' : np.ndarray (hidden_dim,)
            Initial cell state at start of sentence
        '0cx' : np.ndarray (hidden_dim,)
            Zero valued vector to ensure proper decomposition
    """
    def __init__(self,
                 decoder: LinearDecoder,
                 activations: DecomposeArrayDict) -> None:
        self.decoder_w, self.decoder_b = decoder

        self.activations = activations

        self._validate_activation_shapes()

        # Append zero + init cell state to extracted cell states
        self.activations['cx'] = np.vstack((
            activations['0cx'],
            activations['icx'],
            activations['cx']
        ))

    def _decompose(self) -> DecomposeArrayDict:
        raise NotImplementedError

    def decompose(self, append_bias: bool = False, normalize: bool = False) -> DecomposeArrayDict:
        decomposition = self._decompose()

        if append_bias:
            bias_decomposition = self.decompose_bias()
            for key, arr in decomposition.items():
                decomposition[key] = np.vstack((decomposition[key], bias_decomposition))

        if normalize:
            for key, arr in decomposition.items():
                norm_arr = np.log(arr) / np.sum(np.log(arr), axis=0)
                decomposition[key] = norm_arr

        return decomposition

    def decompose_bias(self) -> np.ndarray:
        return np.exp(self.decoder_b)

    def calc_original_logits(self, normalize: bool = False) -> np.ndarray:
        original_logit = np.exp(np.dot(self.activations['hx'], self.decoder_w.T) + self.decoder_b)

        if normalize:
            original_logit /= original_logit.sum()

        return original_logit

    def _validate_activation_shapes(self) -> None:
        decoder_shape = self.decoder_w.shape[1]
        final_output_gate_shape = self.activations['o_g'].shape[0]
        forget_gates_shape = self.activations['f_g'].shape
        init_cell_shape = self.activations['icx'].shape
        cell_states_shape = self.activations['cx'].shape

        assert decoder_shape == final_output_gate_shape, \
            f'Decoder != o_g: {decoder_shape} != {final_output_gate_shape}'
        assert final_output_gate_shape == cell_states_shape[1], \
            f'o_g != cx: {final_output_gate_shape} != {cell_states_shape[1]}'
        assert init_cell_shape[0] == cell_states_shape[1], \
            f'init cx != cx: {init_cell_shape[0]} != {cell_states_shape[1]}'
        assert forget_gates_shape == cell_states_shape, \
            f'f_g != cx: {forget_gates_shape} != {cell_states_shape}'
