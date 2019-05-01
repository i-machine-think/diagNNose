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
    def __init__(self, *args: Any) -> None:
        super().__init__(*args)

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
    def _decompose(self,
                   start: int, stop: int,
                   decompose_o: bool = False,
                   rel_interactions: Optional[List[str]] = None,
                   irrel_interactions: Optional[List[str]] = None,
                   bias_always_irrel: bool = False
                   ) -> DecomposeArrayDict:
        """ Main loop for the contextual decomposition.

        Parameters
        ----------
        start : int
            Starting index of the relevant subphrase
        stop : int
            Stopping index of the relevant subphrase. This stop index is
            not included in the subphrase range, similar to range().
        decompose_o : bool, optional
            Toggles decomposition of the output gate. Defaults to False.
        rel_interactions : List[str], optional
            Indicates the interactions that are part of the relevant
            decomposition. Possible interactions are: rel-rel, rel-b and
            rel-irrel.
        irrel_interactions : List[str], optional
            Indicates the interactions that are part of the irrelevant
            decomposition. Possible interactions are: irrel-irrel,
            irrel-b and rel-irrel.
        bias_always_irrel : bool, optional
            Toggles whether the bias-bias interaction should always be
            added to the irrelevant decomposition. Defaults to false,
            indicating that bias-bias interactions inside the subphrase
            range are added to the relevant decomposition.
        """
        self.rel_interactions = set(rel_interactions or {'rel-rel', 'rel-irrel', 'rel-b'})
        assert len(self.rel_interactions.intersection({'irrel-irrel', 'irrel-b'})) == 0, \
            'irrel-irrel and irrel-b can\'t be part of rel interactions'

        weight, bias = self._get_model_weights()
        self._reset_decompositions(start)

        for layer in range(self.model.num_layers):
            for i in range(start, self.slen):
                self.calc_activations(layer, i, start, stop, weight)

                self.add_forget_decomposition(layer, i, bias[layer, 'f'])

                self.add_input_decomposition(layer, i, start, stop, bias_always_irrel,
                                             bias[layer, 'i'], bias[layer, 'g'])

                self.add_output_decomposition(layer, i, decompose_o, bias[layer, 'o'])

        self._assert_decomposition()

        return self.decompositions

    def calc_activations(self,
                         layer: int, i: int, start: int, stop: int,
                         weight: PartialArrayDict) -> None:
        """ Recalculates the decomposed model activations.

        Input is either the word embedding in layer 0, or the beta/gamma
        decomposition of the hidden state in the previous layer.
        """
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
            if start < 0:
                prev_rel_h = self.activation_dict[layer, 'ihx'][0, 0]
                prev_irrel_h = np.zeros(self.model.hidden_size, dtype=np.float32)
            else:
                prev_rel_h = np.zeros(self.model.hidden_size, dtype=np.float32)
                prev_irrel_h = self.activation_dict[layer, 'ihx'][0, 0]

        self.activations['rel_i'] = weight[layer, 'hi'] @ prev_rel_h + weight[layer, 'ii'] @ rel_input
        self.activations['rel_g'] = weight[layer, 'hg'] @ prev_rel_h + weight[layer, 'ig'] @ rel_input
        self.activations['rel_f'] = weight[layer, 'hf'] @ prev_rel_h + weight[layer, 'if'] @ rel_input
        self.activations['rel_o'] = weight[layer, 'ho'] @ prev_rel_h + weight[layer, 'io'] @ rel_input

        self.activations['irrel_i'] = weight[layer, 'hi'] @ prev_irrel_h + weight[layer, 'ii'] @ irrel_input
        self.activations['irrel_g'] = weight[layer, 'hg'] @ prev_irrel_h + weight[layer, 'ig'] @ irrel_input
        self.activations['irrel_f'] = weight[layer, 'hf'] @ prev_irrel_h + weight[layer, 'if'] @ irrel_input
        self.activations['irrel_o'] = weight[layer, 'ho'] @ prev_irrel_h + weight[layer, 'io'] @ irrel_input

    def add_forget_decomposition(self, layer: int, i: int, bias_f: np.ndarray) -> None:
        """ Calculates the forget gate decomposition, Equation (14) of the paper. """

        rel_contrib_f, irrel_contrib_f, bias_contrib_f = \
            decomp_three(self.activations['rel_f'], self.activations['irrel_f'], bias_f, sigmoid)

        if i > 0:
            prev_rel_c = self.decompositions['relevant_c'][layer][i - 1]
            prev_irrel_c = self.decompositions['irrelevant_c'][layer][i - 1]
        else:
            prev_rel_c = np.zeros(self.model.hidden_size, dtype=np.float32)
            prev_irrel_c = self.activation_dict[layer, 'icx'][0, 0]

        self.add_interactions(layer, i,
                              rel_contrib_f, prev_rel_c,
                              irrel_contrib_f, prev_irrel_c,
                              bias_contrib_f)

    def add_input_decomposition(self,
                                layer: int, i: int, start: int, stop: int, bias_always_irrel: bool,
                                bias_i: np.ndarray, bias_g: np.ndarray) -> None:
        """ Calculates the input gate decomposition, Equation (17) of the paper. """

        rel_contrib_i, irrel_contrib_i, bias_contrib_i = \
            decomp_three(self.activations['rel_i'], self.activations['irrel_i'], bias_i, sigmoid)
        rel_contrib_g, irrel_contrib_g, bias_contrib_g = \
            decomp_three(self.activations['rel_g'], self.activations['irrel_g'], bias_g, np.tanh)

        self.add_interactions(layer, i,
                              rel_contrib_i, rel_contrib_g,
                              irrel_contrib_i, irrel_contrib_g,
                              bias_contrib_i, bias_contrib_g)

        if start <= i < stop and not bias_always_irrel:
            self.decompositions['relevant_c'][layer][i] += bias_contrib_i * bias_contrib_g
        else:
            self.decompositions['irrelevant_c'][layer][i] += bias_contrib_i * bias_contrib_g

    def add_output_decomposition(self,
                                 layer: int, i: int, decompose_o: bool,
                                 bias_o: np.ndarray) -> None:
        """ Calculates the output gate decomposition, Equation (23) of the paper.

        As stated in the paper, output decomposition is not always beneficial
        and can therefore be toggled off.
        """

        rel_h, irrel_h = decomp_tanh_two(self.decompositions['relevant_c'][layer][i],
                                         self.decompositions['irrelevant_c'][layer][i])
        rel_o, irrel_o = self.activations['rel_o'], self.activations['irrel_o']

        if decompose_o:
            rel_contrib_o, irrel_contrib_o, bias_contrib_o = \
                decomp_three(rel_o, irrel_o, bias_o, sigmoid)

            self.add_interactions(layer, i,
                                  rel_contrib_o, rel_h,
                                  irrel_contrib_o, irrel_h,
                                  bias_contrib_o,
                                  rel_decomp_name='relevant_h', irrel_decomp_name='irrelevant_h')
        else:
            o = sigmoid(rel_o + irrel_o + bias_o)
            self.decompositions['relevant_h'][layer][i] = o * rel_h
            self.decompositions['irrelevant_h'][layer][i] = o * irrel_h

    def add_interactions(self,
                         layer: int, i: int,
                         rel_term1: np.ndarray, rel_term2: np.ndarray,
                         irrel_term1: np.ndarray, irrel_term2: np.ndarray,
                         bias_term1: np.ndarray, bias_term2: Optional[np.ndarray] = None,
                         rel_decomp_name: str = 'relevant_c',
                         irrel_decomp_name: str = 'irrelevant_c') -> None:
        """ Allows for interactions to be grouped differently than as specified in the paper. """
        all_interactions = {'rel-rel', 'rel-b', 'irrel-irrel', 'irrel-b', 'rel-irrel'}
        irrel_interactions = all_interactions - self.rel_interactions

        for decomp_name, interactions in ((rel_decomp_name, self.rel_interactions),
                                          (irrel_decomp_name, irrel_interactions)):
            for interaction in interactions:
                if interaction == 'rel-rel':
                    self.decompositions[decomp_name][layer][i] += rel_term1 * rel_term2
                elif interaction == 'rel-b':
                    if bias_term2 is not None:
                        self.decompositions[decomp_name][layer][i] += rel_term1 * bias_term2
                    self.decompositions[decomp_name][layer][i] += rel_term2 * bias_term1
                elif interaction == 'irrel-irrel':
                    self.decompositions[decomp_name][layer][i] += irrel_term1 * irrel_term2
                elif interaction == 'irrel-b':
                    if bias_term2 is not None:
                        self.decompositions[decomp_name][layer][i] += irrel_term1 * bias_term2
                    self.decompositions[decomp_name][layer][i] += irrel_term2 * bias_term1
                elif interaction == 'rel-irrel':
                    self.decompositions[decomp_name][layer][i] += rel_term1 * irrel_term2
                    self.decompositions[decomp_name][layer][i] += rel_term2 * irrel_term1
                else:
                    raise ValueError('Interaction type not understood')

    def _reset_decompositions(self, start: int) -> None:
        hidden_size = self.model.hidden_size
        num_layers = self.model.num_layers

        for rel_i in self.rel_interactions:
            if rel_i == 'rel-rel':
                self.decompositions[rel_decomp_name][layer][i] += rel_term1 * rel_term2
            elif rel_i == 'rel-b':
                if bias_term2 is not None:
                    self.decompositions[rel_decomp_name][layer][i] += rel_term1 * bias_term2
                self.decompositions[rel_decomp_name][layer][i] += rel_term2 * bias_term1
            elif rel_i == 'rel-irrel':
                self.decompositions[rel_decomp_name][layer][i] += rel_term1 * irrel_term2
                self.decompositions[rel_decomp_name][layer][i] += rel_term2 * irrel_term1
            else:
                raise ValueError('Interaction type not understood')

        for rel_i in self.irrel_interactions:
            if rel_i == 'irrel-irrel':
                self.decompositions[irrel_decomp_name][layer][i] += irrel_term1 * irrel_term2
            elif rel_i == 'irrel-b':
                if bias_term2 is not None:
                    self.decompositions[irrel_decomp_name][layer][i] += irrel_term1 * bias_term2
                self.decompositions[irrel_decomp_name][layer][i] += irrel_term2 * bias_term1
            elif rel_i == 'rel-irrel':
                self.decompositions[irrel_decomp_name][layer][i] += irrel_term1 * rel_term2
                self.decompositions[irrel_decomp_name][layer][i] += irrel_term2 * rel_term1
            else:
                raise ValueError('Interaction type not understood')

        # All indices until the start position of the relevant token won't yield any contribution,
        # so we can directly set the cell/hidden states that have been computed already.
        for l in range(num_layers):
            for i in range(start):
                self.decompositions['irrelevant_c'][l][i] = self.activation_dict[(l, 'cx')][0][i]
                self.decompositions['irrelevant_h'][l][i] = self.activation_dict[(l, 'hx')][0][i]

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
