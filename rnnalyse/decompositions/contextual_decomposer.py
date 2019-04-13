from typing import Callable, Tuple

import numpy as np
from overrides import overrides
from scipy.special import expit as sigmoid

from rnnalyse.typedefs.activations import DecomposeArrayDict, PartialArrayDict

from .base_decomposer import BaseDecomposer


class ContextualDecomposer(BaseDecomposer):
    """ Implementation of the LSTM decomposition method described in:

    Murdoch et al., Beyond Word Importance: Contextual Decomposition to
    Extract Interactions from LSTMs (2017)
    https://arxiv.org/pdf/1801.05453.pdf

    Code was partly copied from and inspired by:
    https://github.com/jamie-murdoch/ContextualDecomposition

    Inherits and uses functions from BaseDecomposer.
    """

    @overrides
    def _decompose(self, start: int, stop: int, decompose_o: bool = False) -> DecomposeArrayDict:
        weight, bias = self._get_model_weights()

        word_vecs = self.activation_dict[0, 'emb'][0]
        # TODO: Think about how only a subsentence can be decomposed without having to calculate all
        # irrelevant cell states beforehand, i.e. read those from a_reader
        slen = word_vecs.shape[0]
        hidden_size = self.model.hidden_size
        num_layers = self.model.num_layers

        relevant_c = np.zeros((num_layers, slen, hidden_size), dtype=np.float32)
        irrelevant_c = np.zeros((num_layers, slen, hidden_size), dtype=np.float32)
        relevant_h = np.zeros((num_layers, slen, hidden_size), dtype=np.float32)
        irrelevant_h = np.zeros((num_layers, slen, hidden_size), dtype=np.float32)

        for layer in range(num_layers):
            rel_input = relevant_h[layer-1]
            irrel_input = irrelevant_h[layer-1]

            for i in range(slen):
                if i > 0:
                    prev_rel_h = relevant_h[layer][i-1]
                    prev_rel_c = relevant_c[layer][i-1]
                    prev_irrel_h = irrelevant_h[layer][i-1]
                    prev_irrel_c = irrelevant_c[layer][i-1]
                else:
                    prev_rel_h = np.zeros(hidden_size, dtype=np.float32)
                    prev_rel_c = np.zeros(hidden_size, dtype=np.float32)
                    prev_irrel_h = self.activation_dict[layer, 'ihx'][0, 0]
                    prev_irrel_c = self.activation_dict[layer, 'icx'][0, 0]

                if layer == 0 and start <= i < stop:
                    rel_input = word_vecs
                    irrel_input = np.zeros((slen, hidden_size), dtype=np.float32)
                elif layer == 0:
                    rel_input = np.zeros((slen, hidden_size), dtype=np.float32)
                    irrel_input = word_vecs

                rel_i = weight[layer, 'hi'] @ prev_rel_h + weight[layer, 'ii'] @ rel_input[i]
                rel_g = weight[layer, 'hg'] @ prev_rel_h + weight[layer, 'ig'] @ rel_input[i]
                rel_f = weight[layer, 'hf'] @ prev_rel_h + weight[layer, 'if'] @ rel_input[i]
                rel_o = weight[layer, 'ho'] @ prev_rel_h + weight[layer, 'io'] @ rel_input[i]

                irrel_i = weight[layer, 'hi'] @ prev_irrel_h + weight[layer, 'ii'] @ irrel_input[i]
                irrel_g = weight[layer, 'hg'] @ prev_irrel_h + weight[layer, 'ig'] @ irrel_input[i]
                irrel_f = weight[layer, 'hf'] @ prev_irrel_h + weight[layer, 'if'] @ irrel_input[i]
                irrel_o = weight[layer, 'ho'] @ prev_irrel_h + weight[layer, 'io'] @ irrel_input[i]

                # INPUT DECOMPOSITION
                rel_contrib_i, irrel_contrib_i, bias_contrib_i = decomp_three(rel_i, irrel_i,
                                                                              bias[layer, 'i'],
                                                                              sigmoid)
                rel_contrib_g, irrel_contrib_g, bias_contrib_g = decomp_three(rel_g, irrel_g,
                                                                              bias[layer, 'g'],
                                                                              tanh)

                relevant_c[layer][i] = rel_contrib_i * (
                        rel_contrib_g + bias_contrib_g) + bias_contrib_i * rel_contrib_g
                irrelevant_c[layer][i] = irrel_contrib_i * (
                        rel_contrib_g + irrel_contrib_g + bias_contrib_g) + (
                        rel_contrib_i + bias_contrib_i) * irrel_contrib_g

                # FORGET DECOMPOSITION
                rel_contrib_f, irrel_contrib_f, bias_contrib_f = decomp_three(rel_f, irrel_f,
                                                                              bias[layer, 'f'],
                                                                              sigmoid)
                relevant_c[layer][i] += rel_contrib_f * prev_rel_c
                irrelevant_c[layer][i] += ((rel_contrib_f + irrel_contrib_f + bias_contrib_f) *
                                           prev_irrel_c + irrel_contrib_f * prev_rel_c)

                # BIAS DECOMPOSITION
                if start <= i < stop and layer == 0:
                    relevant_c[layer][i] += bias_contrib_i * bias_contrib_g
                    relevant_c[layer][i] += bias_contrib_f * prev_rel_c
                else:
                    irrelevant_c[layer][i] += bias_contrib_i * bias_contrib_g
                    irrelevant_c[layer][i] += bias_contrib_f * prev_rel_c

                new_rel_h, new_irrel_h = decomp_tanh_two(relevant_c[layer][i], irrelevant_c[layer][i])

                # OUTPUT DECOMPOSITION
                if decompose_o:
                    rel_contrib_o, irrel_contrib_o, bias_contrib_o = decomp_three(rel_o, irrel_o, bias[layer, 'o'], sigmoid)
                    relevant_h[layer][i] = new_rel_h * (rel_contrib_o + bias_contrib_o)
                    irrelevant_h[layer][i] = new_rel_h * irrel_contrib_o + new_irrel_h * (
                            rel_contrib_o + irrel_contrib_o + bias_contrib_o)
                else:
                    o = sigmoid(rel_o + irrel_o + bias[layer, 'o'])
                    relevant_h[layer][i] = o * new_rel_h
                    irrelevant_h[layer][i] = o * new_irrel_h

        self._assert_decomposition(relevant_h, irrelevant_h)

        return {
            'relevant_h': relevant_h,
            'irrelevant_h': irrelevant_h,
            'relevant_c': relevant_c,
            'irrelevant_c': irrelevant_c,
        }

    def _get_model_weights(self) -> Tuple[PartialArrayDict, PartialArrayDict]:
        weight: PartialArrayDict = {}
        bias: PartialArrayDict = {}

        for layer in range(self.model.num_layers):
            for name in ['ii', 'if', 'ig', 'io']:
                weight[layer, name] = self.model.weight[layer][name].detach().numpy()
                bias[layer, name[1]] = (self.model.bias[layer][f'i{name[1]}'].detach().numpy()
                                        + self.model.bias[layer][f'h{name[1]}'].detach().numpy())
            for name in ['hi', 'hf', 'hg', 'ho']:
                weight[layer, name] = self.model.weight[layer][name].detach().numpy()

        return weight, bias

    def _assert_decomposition(self, relevant_h: np.ndarray, irrelevant_h: np.ndarray) -> None:
        final_hidden_state = self.get_final_activations((self.toplayer, 'hx'))
        original_score = final_hidden_state[0] @ self.decoder_w.T

        decomposed_score = relevant_h[-1, -1] @ self.decoder_w.T
        decomposed_score += irrelevant_h[-1, -1] @ self.decoder_w.T

        # Sanity check: scores + irrel_scores should equal the LSTM's output minus bias
        assert np.allclose(original_score, decomposed_score, rtol=1e-4), \
            f'Decomposed score does not match: {original_score} vs {decomposed_score}'


# Activation linearizations as described in chapter 3.2.2
def decomp_three(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                 activation: Callable[[np.ndarray], np.ndarray]
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ac = activation(a + c)
    bc = activation(b + c)
    abc = activation(a + b + c)

    c_contrib = activation(c)
    a_contrib = 0.5 * ((ac - c_contrib) + (abc - bc))
    b_contrib = 0.5 * ((bc - c_contrib) + (abc - ac))

    return a_contrib, b_contrib, c_contrib


def decomp_tanh_two(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    atanh = np.tanh(a)
    btanh = np.tanh(b)
    abtanh = np.tanh(a + b)

    return 0.5 * (atanh + (abtanh - btanh)), 0.5 * (btanh + (abtanh - atanh))


def tanh(arr: np.ndarray) -> np.ndarray:
    return np.tanh(arr)
