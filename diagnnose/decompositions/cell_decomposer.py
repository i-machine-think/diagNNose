from typing import Dict

import numpy as np
import torch
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
        self._append_init_states()

        return {"beta": self.calc_beta(), "gamma": self.calc_gamma()}

    def calc_beta(self) -> np.ndarray:
        """ Calculates the beta term of the paper

        First entry is the contribution of the initial cell state.

        Returns
        -------
        beta : np.ndarray (num_tokens+1, num_classes)
        """
        beta = self._decompose_cell_states(
            self.activation_dict[self.model.top_layer, "cx"]
        )

        return beta

    def calc_gamma(self) -> np.ndarray:
        """ Calculates the gamma term of the paper

        First entry is the contribution of the initial cell state.

        Returns
        -------
        gamma : np.ndarray (num_tokens+1, num_classes)
        """

        gated_cells = np.zeros(self.activation_dict[self.model.top_layer, "cx"].shape)
        for i in range(self.activation_dict[self.model.top_layer, "cx"].shape[1]):
            forget_product = np.prod(
                self.activation_dict[self.model.top_layer, "f_g"][:, i:], axis=1
            )
            gated_cells[:, i] = (
                self.activation_dict[self.model.top_layer, "cx"][:, i] * forget_product
            )

        gamma = self._decompose_cell_states(gated_cells)

        return gamma

    def _decompose_cell_states(self, cell_states: np.ndarray) -> np.ndarray:
        cell_diffs = np.tanh(cell_states[:, 1:]) - np.tanh(cell_states[:, :-1])

        final_output_gate = self.get_final_activations((self.model.top_layer, "o_g"))
        final_output_gate = np.expand_dims(final_output_gate, axis=1)

        decomposed_h = final_output_gate * cell_diffs
        decomposition = np.exp(np.ma.dot(decomposed_h, self.decoder_w.t()))

        self._assert_decomposition(decomposition)

        return decomposition.data

    def _assert_decomposition(self, decomposition: np.ndarray) -> None:
        original_logits = self.calc_original_logits()
        decomposed_logit = np.prod(decomposition, axis=1) * self.decompose_bias()

        # Cell state array has 2 init states appended to it, hence the offset
        final_cell_state = self.get_final_activations(
            (self.model.top_layer, "cx"), offset=2
        )
        final_output_gate = self.get_final_activations((self.model.top_layer, "o_g"))
        final_hidden_state = self.get_final_activations((self.model.top_layer, "hx"))

        reconstructed_h = final_output_gate * np.tanh(final_cell_state)

        np.testing.assert_array_almost_equal(
            final_hidden_state,
            reconstructed_h,
            err_msg="Reconstructed state h_T not equal to provided hidden state",
        )
        assert np.allclose(
            original_logits, decomposed_logit, rtol=1e-4
        ), f"Decomposed logits not equal to original\n{original_logits}\n\n{decomposed_logit}"

    def _append_init_states(self) -> None:
        """Append icx/ihx to cx/hx activations."""
        for layer, name in self.activation_dict:
            if name.startswith("i") and name[1:] in ["cx", "hx"]:
                cell_type = name[1:]
                if (layer, cell_type) in self.activation_dict:
                    self.activation_dict[(layer, cell_type)] = torch.cat(
                        (
                            self.activation_dict[(layer, name)].unsqueeze(1),
                            self.activation_dict[(layer, cell_type)],
                        ),
                        dim=1,
                    )

                    if cell_type == "hx" and layer == self.model.top_layer:
                        self.final_index += 1

                    # 0cx activations should be concatenated in front of the icx activations.
                    if (layer, f"0{cell_type}") in self.activation_dict:
                        self.activation_dict[(layer, cell_type)] = torch.cat(
                            (
                                self.activation_dict[
                                    (layer, f"0{cell_type}")
                                ].unsqueeze(1),
                                self.activation_dict[(layer, cell_type)],
                            ),
                            dim=1,
                        )

                        if cell_type == "hx" and layer == self.model.top_layer:
                            self.final_index += 1

    @overrides
    def _validate_activation_shapes(self) -> None:
        # (num_classes, hidden_size)
        decoder_shape = self.decoder_w.shape

        # (batch_size, max_sen_len, hidden_size)
        output_gate_shape = self.activation_dict[self.model.top_layer, "o_g"].shape

        # (batch_size, max_sen_len, hidden_size)
        forget_gates_shape = self.activation_dict[self.model.top_layer, "f_g"].shape

        # (batch_size, 1, hidden_size)
        init_cell_shape = self.activation_dict[self.model.top_layer, "icx"].shape

        # (batch_size, max_sen_len, hidden_size)
        cell_states_shape = self.activation_dict[self.model.top_layer, "cx"].shape

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
