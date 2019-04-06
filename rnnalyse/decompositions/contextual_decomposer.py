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
    def _decompose(self, start: int, stop: int) -> DecomposeArrayDict:
        weight, bias = self._get_model_weights()

        word_vecs = self.activation_dict[0, 'emb']
        slen = word_vecs.size(0)

        relevant_c = irrelevant_c = np.zeros((self.model.num_layers, slen, self.model.hidden_size))
        relevant_h = irrelevant_h = np.zeros((self.model.num_layers, slen, self.model.hidden_size))

        for layer in range(self.model.num_layers):
            if layer == 0:
                rel_input = word_vecs
                irrel_input = word_vecs
            else:
                rel_input = relevant_h[layer-1]
                irrel_input = irrelevant_h[layer-1]

            for i in range(slen):
                if i > 0:
                    prev_rel_h = relevant_h[layer][i - 1]
                    prev_irrel_h = irrelevant_h[layer][i - 1]
                else:
                    prev_rel_h = self.activation_dict[layer, 'icx']
                    prev_irrel_h = self.activation_dict[layer, 'ihx']

                if layer == 0 and start <= i <= stop:
                    rel_input = word_vecs
                    irrel_input = np.zeros((slen, self.model.hidden_size))
                elif layer == 0:
                    rel_input = np.zeros((slen, self.model.hidden_size))
                    irrel_input = word_vecs

                rel_i = weight[layer, 'hi'] @ prev_rel_h + weight[layer, 'ii'] @ rel_input[i]
                rel_g = weight[layer, 'hg'] @ prev_rel_h + weight[layer, 'ig'] @ rel_input[i]
                rel_f = weight[layer, 'hf'] @ prev_rel_h + weight[layer, 'if'] @ rel_input[i]
                rel_o = weight[layer, 'ho'] @ prev_rel_h + weight[layer, 'io'] @ rel_input[i]

                irrel_i = weight[layer, 'hi'] @ prev_irrel_h + weight[layer, 'ii'] @ irrel_input[i]
                irrel_g = weight[layer, 'hg'] @ prev_irrel_h + weight[layer, 'ig'] @ irrel_input[i]
                irrel_f = weight[layer, 'hf'] @ prev_irrel_h + weight[layer, 'if'] @ irrel_input[i]
                irrel_o = weight[layer, 'ho'] @ prev_irrel_h + weight[layer, 'io'] @ irrel_input[i]

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

                # TODO: check how the following influences results
                if start <= i < stop and layer == 0:
                    relevant_c[i] += bias_contrib_i * bias_contrib_g
                else:
                    irrelevant_c[i] += bias_contrib_i * bias_contrib_g

                # TODO: check why i > 0 and if indices match
                if i > 0:
                    rel_contrib_f, irrel_contrib_f, bias_contrib_f = decomp_three(rel_f, irrel_f,
                                                                                  bias[layer, 'f'],
                                                                                  sigmoid)
                    relevant_c[i] += (rel_contrib_f + bias_contrib_f) * relevant_c[i - 1]
                    irrelevant_c[i] += ((rel_contrib_f + irrel_contrib_f + bias_contrib_f) *
                                        irrelevant_c[i - 1] + irrel_contrib_f * relevant_c[i - 1])

                o = sigmoid(weight[layer, 'io'] @ word_vecs[i]
                            + weight[layer, 'ho'] @ prev_rel_h + prev_irrel_h
                            + bias[layer, 'o'])
                # rel_contrib_o, irrel_contrib_o, bias_contrib_o =
                # decomp_three(rel_o, irrel_o, b_o, sigmoid)
                new_rel_h, new_irrel_h = decomp_tanh_two(relevant_c[i], irrelevant_c[i])
                # relevant_h[i] = new_rel_h * (rel_contrib_o + bias_contrib_o)
                # irrelevant_h[i] = new_rel_h * (irrel_contrib_o) + new_irrel_h *
                # (rel_contrib_o + irrel_contrib_o + bias_contrib_o)
                relevant_h[i] = o * new_rel_h
                irrelevant_h[i] = o * new_irrel_h

        # Sanity check: scores + irrel_scores should equal the LSTM's output minus bias
        scores = relevant_h[-1, slen - 1] @ self.decoder_w.T
        irrel_scores = irrelevant_h[-1, slen - 1] @ self.decoder_w.T

        self._assert_decomposition(scores + irrel_scores)

        return {
            'relevant_h': relevant_h,
            'irrelevant_h': irrelevant_h,
        }

    def _get_model_weights(self) -> Tuple[PartialArrayDict, PartialArrayDict]:
        weight: PartialArrayDict = {}
        bias: PartialArrayDict = {}

        for layer in range(self.model.num_layers):
            for name in ['ii', 'if', 'ig', 'io']:
                weight[layer, name] = self.model.weight[layer]['ii'].cpu().numpy()
                bias[layer, name] = (self.model.bias[0][f'i{name[1]}'].cpu().numpy()
                                     + self.model.bias[0][f'h{name[1]}'].cpu().numpy())
            for name in ['hi', 'hf', 'hg', 'ho']:
                weight[layer, name] = self.model.weight[layer]['ii'].cpu().numpy()

        return weight, bias

    def _assert_decomposition(self, decomposed_score: np.ndarray) -> None:
        final_hidden_state = self.get_final_activations((self.layer, 'hx'))
        orig_score = np.ma.dot(final_hidden_state, self.decoder_w.T)

        assert orig_score == decomposed_score, \
            f'Decompose score does not match: {orig_score} vs {decomposed_score}'


# Activation linearizations as described in chapter 3.2.2
def decomp_three(a: np.ndarray, b: np.ndarray, c: np.ndarray,
                 activation: Callable[[np.ndarray], np.ndarray]
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a_contrib = 0.5 * (
                activation(a + c) - activation(c) + activation(a + b + c) - activation(b + c))
    b_contrib = 0.5 * (
                activation(b + c) - activation(c) + activation(a + b + c) - activation(a + c))
    return a_contrib, b_contrib, activation(c)


def decomp_tanh_two(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return 0.5 * (np.tanh(a) + (np.tanh(a + b) - np.tanh(b))), 0.5 * (
                np.tanh(b) + (np.tanh(a + b) - np.tanh(a)))


def tanh(arr: np.ndarray) -> np.ndarray:
    return np.tanh(arr)
