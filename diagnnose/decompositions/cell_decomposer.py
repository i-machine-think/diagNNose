from typing import Dict

import numpy as np
from overrides import overrides

from .base_decomposer import BaseDecomposer


# TODO: Update class with new structure, currently broken.
class CellDecomposer(BaseDecomposer):
    """ Implementation of the LSTM decomposition method described in:

    Murdoch and Szlam, Automatic Rule Extraction from Long Short Term
    Memory Networks (2017) https://arxiv.org/pdf/1702.02540.pdf
    """

    @overrides
    def _decompose(self) -> Dict[str, np.ndarray]:
        return {"beta": self.calc_beta(), "gamma": self.calc_gamma()}

    def calc_beta(self) -> np.ndarray:
        """ Calculates the beta term of the paper

        First entry is the contribution of the initial cell state.

        Returns
        -------
        beta : np.ndarray (num_tokens+1, num_classes)
        """
        beta = self._decompose_cell_states(self.activation_dict[self.toplayer, "cx"])

        return beta

    def calc_gamma(self) -> np.ndarray:
        """ Calculates the gamma term of the paper

        First entry is the contribution of the initial cell state.

        Returns
        -------
        gamma : np.ndarray (num_tokens+1, num_classes)
        """

        gated_cells = np.zeros(self.activation_dict[self.toplayer, "cx"].shape)
        for i in range(self.activation_dict[self.toplayer, "cx"].shape[1]):
            forget_product = np.prod(
                self.activation_dict[self.toplayer, "f_g"][:, i:], axis=1
            )
            gated_cells[:, i] = (
                self.activation_dict[self.toplayer, "cx"][:, i] * forget_product
            )

        gamma = self._decompose_cell_states(gated_cells)

        return gamma

    def _decompose_cell_states(self, cell_states: np.ndarray) -> np.ndarray:
        cell_diffs = np.tanh(cell_states[:, 1:]) - np.tanh(cell_states[:, :-1])

        final_output_gate = self.get_final_activations((self.toplayer, "o_g"))
        final_output_gate = np.expand_dims(final_output_gate, axis=1)

        decomposed_h = final_output_gate * cell_diffs
        decomposition = np.exp(np.ma.dot(decomposed_h, self.decoder_w.t()))

        self._assert_decomposition(decomposition)

        return decomposition.data

    def _assert_decomposition(self, decomposition: np.ndarray) -> None:
        original_logits = self.calc_original_logits()
        decomposed_logit = np.prod(decomposition, axis=1) * self.decompose_bias()

        # Cell state array has 2 init states appended to it, hence the offset
        final_cell_state = self.get_final_activations((self.toplayer, "cx"), offset=2)
        final_output_gate = self.get_final_activations((self.toplayer, "o_g"))
        final_hidden_state = self.get_final_activations((self.toplayer, "hx"))

        reconstructed_h = final_output_gate * np.tanh(final_cell_state)

        np.testing.assert_array_almost_equal(
            final_hidden_state,
            reconstructed_h,
            err_msg="Reconstructed state h_T not equal to provided hidden state",
        )
        assert np.allclose(
            original_logits, decomposed_logit, rtol=1e-4
        ), f"Decomposed logits not equal to original\n{original_logits}\n\n{decomposed_logit}"

    @overrides
    def _validate_activation_shapes(self) -> None:
        # (num_classes, hidden_size)
        decoder_shape = self.decoder_w.shape

        # (batch_size, max_sen_len, hidden_size)
        output_gate_shape = self.activation_dict[self.toplayer, "o_g"].shape

        # (batch_size, max_sen_len, hidden_size)
        forget_gates_shape = self.activation_dict[self.toplayer, "f_g"].shape

        # (batch_size, 1, hidden_size)
        init_cell_shape = self.activation_dict[self.toplayer, "icx"].shape

        # (batch_size, max_sen_len, hidden_size)
        cell_states_shape = self.activation_dict[self.toplayer, "cx"].shape

        assert (
            decoder_shape[1] == output_gate_shape[2]
        ), f"Decoder != o_g: {decoder_shape} != {output_gate_shape[2]}"
        assert (
            output_gate_shape == cell_states_shape == forget_gates_shape
        ), f"o_g != cx: {output_gate_shape} != {cell_states_shape} != {forget_gates_shape}"
        assert (
            init_cell_shape[0] == cell_states_shape[0]
        ), f"init cx != cx: {init_cell_shape[0]} != {cell_states_shape[0]}"
        assert (
            init_cell_shape[2] == cell_states_shape[2]
        ), f"init cx != cx: {init_cell_shape[2]} != {cell_states_shape[2]}"
