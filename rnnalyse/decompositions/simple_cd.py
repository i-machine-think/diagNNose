import numpy as np
from overrides import overrides

from rnnalyse.typedefs.activations import DecomposeArrayDict
from rnnalyse.typedefs.classifiers import LinearDecoder

from .base_decomposer import BaseDecomposer


class SimpleCD(BaseDecomposer):
    """ Implementation of the LSTM decomposition method described in:

    Murdoch and Szlam, Automatic Rule Extraction from Long Short Term
    Memory Networks (2017) https://arxiv.org/pdf/1702.02540.pdf

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
        super().__init__(decoder, activations)

    @overrides
    def _decompose(self) -> DecomposeArrayDict:
        return {
            'beta': self.calc_beta(),
            'gamma': self.calc_gamma(),
        }

    def calc_beta(self) -> np.ndarray:
        """ Calculates the beta term of the paper

        First entry is the contribution of the initial cell state.

        Returns
        -------
        beta : np.ndarray (num_tokens+1, num_classes)
        """
        beta = self._decompose_cell_states(self.activations['cx'])

        return beta

    def calc_gamma(self) -> np.ndarray:
        """ Calculates the gamma term of the paper

        First entry is the contribution of the initial cell state.

        Returns
        -------
        gamma : np.ndarray (num_tokens+1, num_classes)
        """

        gated_cells = np.zeros(self.activations['cx'].shape)
        for i in range(self.activations['cx'].shape[0]):
            gated_cells[i] = self.activations['cx'][i] * np.prod(self.activations['f_g'][i:], axis=0)

        gamma = self._decompose_cell_states(gated_cells)

        return gamma

    def _decompose_cell_states(self, cell_states: np.ndarray) -> np.ndarray:
        cell_diffs = np.tanh(cell_states[1:]) - np.tanh(cell_states[:-1])

        decomposed_h = self.activations['o_g'] * cell_diffs
        decomposition = np.exp(np.dot(decomposed_h, self.decoder_w.T))

        self._assert_decomposition(decomposition)

        return decomposition

    def _assert_decomposition(self, decomposition: np.ndarray) -> None:
        original_logits = self.calc_original_logits()
        reconstructed_h = self.activations['o_g'] * np.tanh(self.activations['cx'][-1])

        decomposed_logit = np.prod(decomposition, axis=0) * self.decompose_bias()

        np.testing.assert_array_almost_equal(
            self.activations['hx'], reconstructed_h,
            err_msg='Reconstructed state h_T not equal to provided hidden state')
        assert np.allclose(original_logits, decomposed_logit, rtol=1e-4), \
            f'Decomposed logits not equal to original\n{original_logits}\n{decomposed_logit}'
