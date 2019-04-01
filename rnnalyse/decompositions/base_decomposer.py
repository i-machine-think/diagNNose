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
    final_index : np.ndarray
        1-d numpy array with index of final element of a batch element.
        Due to masking for sentences of uneven length the final index
        can differ between batch elements.
    """
    def __init__(self,
                 decoder: LinearDecoder,
                 activations: DecomposeArrayDict,
                 batch_size: int,
                 final_index: np.ndarray) -> None:
        self.decoder_w, self.decoder_b = decoder
        self.batch_size = batch_size
        self.final_index = final_index

        self.activations = activations
        self._validate_activation_shapes()

        # Append zero + init cell state to extracted cell states
        self.activations['cx'] = np.ma.concatenate((
            activations['0cx'],
            activations['icx'],
            activations['cx']
        ), axis=1)

    def _decompose(self) -> DecomposeArrayDict:
        raise NotImplementedError

    def decompose(self, append_bias: bool = False) -> DecomposeArrayDict:
        decomposition = self._decompose()

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
        original_logit = np.exp(np.ma.dot(self.activations['hx'], self.decoder_w.T) + bias)

        if normalize:
            original_logit = (original_logit.T / np.sum(original_logit, axis=1)).T

        return original_logit

    def _validate_activation_shapes(self) -> None:
        decoder_shape = self.decoder_w.shape[1]
        final_output_gate_shape = self.activations['o_g'].shape
        forget_gates_shape = self.activations['f_g'].shape
        init_cell_shape = self.activations['icx'].shape
        cell_states_shape = self.activations['cx'].shape

        assert decoder_shape == final_output_gate_shape[1], \
            f'Decoder != o_g: {decoder_shape} != {final_output_gate_shape[1]}'
        assert final_output_gate_shape[1] == cell_states_shape[2], \
            f'o_g != cx: {final_output_gate_shape[1]} != {cell_states_shape[2]}'
        assert init_cell_shape[0] == cell_states_shape[0], \
            f'init cx != cx: {init_cell_shape[0]} != {cell_states_shape[0]}'
        assert init_cell_shape[2] == cell_states_shape[2], \
            f'init cx != cx: {init_cell_shape[2]} != {cell_states_shape[2]}'
        assert forget_gates_shape == cell_states_shape, \
            f'f_g != cx: {forget_gates_shape} != {cell_states_shape}'
