import numpy as np
from overrides import overrides

from rnnalyse.typedefs.activations import DecomposeArrayDict
from rnnalyse.typedefs.classifiers import LinearDecoder

from .base_decomposer import BaseDecomposer


class SimpleCD(BaseDecomposer):
    """ Implementation of the LSTM decomposition method described in:

    Murdoch and Szlam, Automatic Rule Extraction from Long Short Term
    Memory Networks (2017) https://arxiv.org/pdf/1702.02540.pdf

    Inherits and uses functions from BaseDecomposer.
    """

    @overrides
    def _decompose(self) -> DecomposeArrayDict:
        return {
            'beta': self.calc_beta(),
            # 'gamma': self.calc_gamma(),
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
            forget_product = np.prod(self.activations['f_g'][i:], axis=0)
            gated_cells[i] = self.activations['cx'][i] * forget_product

        gamma = self._decompose_cell_states(gated_cells)

        return gamma

    def _decompose_cell_states(self, cell_states: np.ndarray) -> np.ndarray:
        cell_diffs = np.tanh(cell_states[:, 1:]) - np.tanh(cell_states[:, :-1])

        final_output_gate = np.expand_dims(self.activations['o_g'], 1)
        decomposed_h = final_output_gate * cell_diffs
        decomposition = np.exp(np.ma.dot(decomposed_h, self.decoder_w.T))

        self._assert_decomposition(decomposition)

        return decomposition.data

    def _assert_decomposition(self, decomposition: np.ndarray) -> None:
        original_logits = self.calc_original_logits()
        decomposed_logit = np.prod(decomposition, axis=1) * self.decompose_bias()

        final_cell_state = self.activations['cx'][range(self.batch_size), [self.final_index+2]][0]
        reconstructed_h = self.activations['o_g'] * np.tanh(final_cell_state)

        np.testing.assert_array_almost_equal(
            self.activations['hx'], reconstructed_h,
            err_msg='Reconstructed state h_T not equal to provided hidden state')
        assert np.allclose(original_logits, decomposed_logit, rtol=1e-4), \
            f'Decomposed logits not equal to original\n{original_logits}\n\n{decomposed_logit}'
