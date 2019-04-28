from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from overrides import overrides
from scipy.special import expit as sigmoid

from diagnnose.typedefs.activations import DecomposeArrayDict, PartialArrayDict

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
    def __init__(self, *args: Any,
                 rel_interactions: Optional[List[str]] = None,
                 irrel_interactions: Optional[List[str]] = None,
                 bias_always_irrel: bool = False) -> None:
        super().__init__(*args)

        self.rel_interactions = rel_interactions or ['rel-rel', 'rel-b']
        self.irrel_interactions = irrel_interactions or ['irrel-irrel', 'irrel-b']

        word_vecs = self.activation_dict[0, 'emb'][0]
        hidden_size = self.model.hidden_size
        num_layers = self.model.num_layers

        self.slen = word_vecs.shape[0]
        self.activations: DecomposeArrayDict = {}
        self.decompositions: DecomposeArrayDict = {
            'relevant_c': np.zeros((num_layers, self.slen, hidden_size), dtype=np.float32),
            'irrelevant_c': np.zeros((num_layers, self.slen, hidden_size), dtype=np.float32),
            'relevant_h': np.zeros((num_layers, self.slen, hidden_size), dtype=np.float32),
            'irrelevant_h': np.zeros((num_layers, self.slen, hidden_size), dtype=np.float32)
        }

    @overrides
    def _decompose(self, start: int, stop: int, decompose_o: bool = False) -> DecomposeArrayDict:
        weight, bias = self._get_model_weights()

        for layer in range(self.model.num_layers):
            for i in range(self.slen):
                self.calc_activations(layer, i, start, stop, weight)

                self.add_forget_decomposition(layer, i, bias[layer, 'f'])

                self.add_input_decomposition(layer, i, start, stop,
                                             bias[layer, 'i'], bias[layer, 'g'])

                self.add_output_decomposition(layer, i, decompose_o, bias[layer, 'o'])

        self._assert_decomposition()

        return self.decompositions

    def calc_activations(self,
                         layer: int, i: int, start: int, stop: int,
                         weight: PartialArrayDict) -> None:
        if layer == 0:
            if start <= i < stop:
                rel_input = self.activation_dict[0, 'emb'][0][i]
                irrel_input = np.zeros(self.model.hidden_size, dtype=np.float32)
            else:
                rel_input = np.zeros(self.model.hidden_size, dtype=np.float32)
                irrel_input = self.activation_dict[0, 'emb'][0][i]
        else:
            rel_input = self.decompositions['relevant_h'][layer - 1][i]
            irrel_input = self.decompositions['irrelevant_h'][layer - 1][i]

        if i > 0:
            prev_rel_h = self.decompositions['relevant_h'][layer][i - 1]
            prev_irrel_h = self.decompositions['irrelevant_h'][layer][i - 1]
        else:
            prev_rel_h = np.zeros(self.model.hidden_size, dtype=np.float32)
            prev_irrel_h = self.activation_dict[layer, 'ihx'][0, 0]

        self.activations['rel_i'] = \
            weight[layer, 'hi'] @ prev_rel_h + weight[layer, 'ii'] @ rel_input
        self.activations['rel_g'] = \
            weight[layer, 'hg'] @ prev_rel_h + weight[layer, 'ig'] @ rel_input
        self.activations['rel_f'] = \
            weight[layer, 'hf'] @ prev_rel_h + weight[layer, 'if'] @ rel_input
        self.activations['rel_o'] = \
            weight[layer, 'ho'] @ prev_rel_h + weight[layer, 'io'] @ rel_input

        self.activations['irrel_i'] = \
            weight[layer, 'hi'] @ prev_irrel_h + weight[layer, 'ii'] @ irrel_input
        self.activations['irrel_g'] = \
            weight[layer, 'hg'] @ prev_irrel_h + weight[layer, 'ig'] @ irrel_input
        self.activations['irrel_f'] = \
            weight[layer, 'hf'] @ prev_irrel_h + weight[layer, 'if'] @ irrel_input
        self.activations['irrel_o'] = \
            weight[layer, 'ho'] @ prev_irrel_h + weight[layer, 'io'] @ irrel_input

    def add_forget_decomposition(self, layer: int, i: int, bias_f: np.ndarray) -> None:
        rel_contrib_f, irrel_contrib_f, bias_contrib_f = \
            decomp_three(self.activations['rel_f'], self.activations['irrel_f'], bias_f, sigmoid)

        if i > 0:
            prev_rel_c = self.decompositions['relevant_c'][layer][i - 1]
            prev_irrel_c = self.decompositions['irrelevant_c'][layer][i - 1]
        else:
            prev_rel_c = np.zeros(self.model.hidden_size, dtype=np.float32)
            prev_irrel_c = self.activation_dict[layer, 'icx'][0, 0]

        self.decompositions['relevant_c'][layer][i] += (
                prev_rel_c * (rel_contrib_f + bias_contrib_f)
        )
        self.decompositions['irrelevant_c'][layer][i] += (
                prev_irrel_c * (rel_contrib_f + irrel_contrib_f + bias_contrib_f)
                + irrel_contrib_f * prev_rel_c
        )

    def add_input_decomposition(self,
                                layer: int, i: int, start: int, stop: int,
                                bias_i: np.ndarray, bias_g: np.ndarray) -> None:
        rel_contrib_i, irrel_contrib_i, bias_contrib_i = \
            decomp_three(self.activations['rel_i'], self.activations['irrel_i'], bias_i, sigmoid)
        rel_contrib_g, irrel_contrib_g, bias_contrib_g = \
            decomp_three(self.activations['rel_g'], self.activations['irrel_g'], bias_g, np.tanh)

        self.decompositions['relevant_c'][layer][i] += (
                rel_contrib_i * (rel_contrib_g + bias_contrib_g)
                + rel_contrib_g * bias_contrib_i
        )
        self.decompositions['irrelevant_c'][layer][i] += (
                irrel_contrib_i * (rel_contrib_g + irrel_contrib_g + bias_contrib_g)
                + irrel_contrib_g * (rel_contrib_i + bias_contrib_i)
        )

        if start <= i < stop:
            self.decompositions['relevant_c'][layer][i] += bias_contrib_i * bias_contrib_g
        else:
            self.decompositions['irrelevant_c'][layer][i] += bias_contrib_i * bias_contrib_g

    def add_output_decomposition(self,
                                 layer: int, i: int, decompose_o: bool,
                                 bias_o: np.ndarray) -> None:
        rel_h, irrel_h = decomp_tanh_two(self.decompositions['relevant_c'][layer][i],
                                         self.decompositions['irrelevant_c'][layer][i])
        rel_o, irrel_o = self.activations['rel_o'], self.activations['irrel_o']

        if decompose_o:
            rel_contrib_o, irrel_contrib_o, bias_contrib_o = \
                decomp_three(rel_o, irrel_o, bias_o, sigmoid)

            self.decompositions['relevant_h'][layer][i] = (
                    rel_h * (rel_contrib_o + bias_contrib_o)
            )
            self.decompositions['irrelevant_h'][layer][i] = (
                    rel_h * irrel_contrib_o
                    + irrel_h * (rel_contrib_o + irrel_contrib_o + bias_contrib_o)
            )
        else:
            o = sigmoid(rel_o + irrel_o + bias_o)
            self.decompositions['relevant_h'][layer][i] = o * rel_h
            self.decompositions['irrelevant_h'][layer][i] = o * irrel_h

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

    def _assert_decomposition(self) -> None:
        final_hidden_state = self.get_final_activations((self.toplayer, 'hx'))
        original_score = final_hidden_state[0] @ self.decoder_w.T

        decomposed_score = (self.decompositions['relevant_h'][-1, -1] +
                            self.decompositions['irrelevant_h'][-1, -1]) @ self.decoder_w.T

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
