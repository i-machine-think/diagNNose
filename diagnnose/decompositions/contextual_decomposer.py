from typing import Any, Callable, List, Optional, Set, Tuple

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

        self.weight: PartialArrayDict = {}
        self.bias: PartialArrayDict = {}
        self.activations: DecomposeArrayDict = {}
        self.decompositions: DecomposeArrayDict = {}

        self.rel_interactions: Set[str] = set()
        self.irrel_interactions: Set[str] = set()

    @overrides
    def _decompose(self, start: int, stop: int,
                   rel_interactions: Optional[List[str]] = None,
                   decompose_o: bool = False,
                   bias_bias_only_in_phrase: bool = True,
                   only_source_rel: bool = False,
                   input_never_rel: bool = False,
                   use_extracted_activations: bool = False,
                   validate: bool = True,
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
        bias_bias_only_in_phrase : bool, optional
            Toggles whether the bias-bias interaction should only be
            added when inside the relevant phrase. Defaults to True,
            indicating that only bias-bias interactions inside the
            subphrase range are added to the relevant decomposition.
        only_source_rel : bool, optional
            Relates to rel-irrel interactions. If set to true, only
            irrel_gate-rel_source interactions will be added to rel,
            similar to LRP (Arras et al., 2017).
        input_never_rel : bool, optional
            Never add the Wx input to the rel part, useful when only
            investigating the model biases. Defaults to False.
        use_extracted_activations : bool, optional
            Allows previously extracted activations to be used to avoid
            unnecessary recomputations of those activations.
            Defaults to False.
        validate : bool, optional
            Toggles decomposition validation, defaults to True.

        Returns
        -------
        scores : DecomposeArrayDict
            Dictionary with keys `relevant` and `irrelevant`, containing
            the decomposed scores for the earlier provided decoder.
        """
        self._set_rel_interactions(rel_interactions)

        start_index = max(0, start) if use_extracted_activations else 0
        slen = self.activation_dict[0, 'emb'].shape[1]

        self._set_model_weights()
        self._reset_decompositions(start, slen, use_extracted_activations)

        self.bias_bias_only_in_phrase = bias_bias_only_in_phrase
        self.only_source_rel = only_source_rel

        for layer in range(self.model.num_layers):
            for i in range(start_index, slen):
                inside_phrase = start <= i < stop

                self._calc_activations(layer, i, start, inside_phrase, input_never_rel)

                self._add_forget_decomposition(layer, i, start)

                self._add_input_decomposition(layer, i, inside_phrase)

                self._add_output_decomposition(layer, i, decompose_o)

        scores = self._calc_scores()

        if validate:
            self._validate_decomposition(scores)

        return scores

    def _calc_activations(self, layer: int, i: int, start: int,
                          inside_phrase: bool, input_never_rel: bool) -> None:
        """ Recalculates the decomposed model activations.

        Input is either the word embedding in layer 0, or the beta/gamma
        decomposition of the hidden state in the previous layer.
        """
        if layer == 0:
            if inside_phrase and not input_never_rel:
                rel_input = self.activation_dict[0, 'emb'][0][i]
                irrel_input = np.zeros(self.model.hidden_size, dtype=np.float32)
            else:
                rel_input = np.zeros(self.model.hidden_size, dtype=np.float32)
                irrel_input = self.activation_dict[0, 'emb'][0][i]
        else:
            rel_input = self.decompositions['relevant_h'][layer - 1][i]
            irrel_input = self.decompositions['irrelevant_h'][layer - 1][i]

        prev_rel_h, prev_irrel_h = self._get_prev_cells(layer, i, start, 'h')

        w = self.weight
        self.activations['rel_i'] = w[layer, 'hi'] @ prev_rel_h + w[layer, 'ii'] @ rel_input
        self.activations['rel_g'] = w[layer, 'hg'] @ prev_rel_h + w[layer, 'ig'] @ rel_input
        self.activations['rel_f'] = w[layer, 'hf'] @ prev_rel_h + w[layer, 'if'] @ rel_input
        self.activations['rel_o'] = w[layer, 'ho'] @ prev_rel_h + w[layer, 'io'] @ rel_input

        self.activations['irrel_i'] = w[layer, 'hi'] @ prev_irrel_h + w[layer, 'ii'] @ irrel_input
        self.activations['irrel_g'] = w[layer, 'hg'] @ prev_irrel_h + w[layer, 'ig'] @ irrel_input
        self.activations['irrel_f'] = w[layer, 'hf'] @ prev_irrel_h + w[layer, 'if'] @ irrel_input
        self.activations['irrel_o'] = w[layer, 'ho'] @ prev_irrel_h + w[layer, 'io'] @ irrel_input

    def _add_forget_decomposition(self, layer: int, i: int, start: int) -> None:
        """ Calculates the forget gate decomposition, Equation (14) of the paper. """

        rel_contrib_f, irrel_contrib_f, bias_contrib_f = \
            decomp_three(self.activations['rel_f'], self.activations['irrel_f'],
                         self.bias[layer, 'f'], sigmoid)

        prev_rel_c, prev_irrel_c = self._get_prev_cells(layer, i, start, 'c')

        self._add_interactions(layer, i,
                               rel_contrib_f, prev_rel_c,
                               irrel_contrib_f, prev_irrel_c,
                               bias_contrib_f)

    def _add_input_decomposition(self, layer: int, i: int, inside_phrase: bool) -> None:
        """ Calculates the input gate decomposition, Equation (17) of the paper. """

        rel_contrib_i, irrel_contrib_i, bias_contrib_i = \
            decomp_three(self.activations['rel_i'], self.activations['irrel_i'],
                         self.bias[layer, 'i'], sigmoid)
        rel_contrib_g, irrel_contrib_g, bias_contrib_g = \
            decomp_three(self.activations['rel_g'], self.activations['irrel_g'],
                         self.bias[layer, 'g'], np.tanh)

        self._add_interactions(layer, i,
                               rel_contrib_i, rel_contrib_g,
                               irrel_contrib_i, irrel_contrib_g,
                               bias_contrib_i, bias_contrib_g,
                               inside_phrase=inside_phrase)

    def _add_output_decomposition(self, layer: int, i: int, decompose_o: bool) -> None:
        """ Calculates the output gate decomposition, Equation (23).

        As stated in the paper, output decomposition is not always
        beneficial and can therefore be toggled off.
        """

        rel_c, irrel_c = decomp_tanh_two(self.decompositions['relevant_c'][layer][i],
                                         self.decompositions['irrelevant_c'][layer][i])
        rel_o, irrel_o = self.activations['rel_o'], self.activations['irrel_o']

        if decompose_o:
            rel_contrib_o, irrel_contrib_o, bias_contrib_o = \
                decomp_three(rel_o, irrel_o, self.bias[layer, 'o'], sigmoid)

            self._add_interactions(layer, i,
                                   rel_contrib_o, rel_c,
                                   irrel_contrib_o, irrel_c,
                                   bias_contrib_o,
                                   cell_type='h')
        else:
            o = sigmoid(rel_o + irrel_o + self.bias[layer, 'o'])
            self.decompositions['relevant_h'][layer][i] = o * rel_c
            self.decompositions['irrelevant_h'][layer][i] = o * irrel_c

    def _get_prev_cells(self, layer: int, i: int, start: int,
                        cell_type: str) -> Tuple[np.ndarray, np.ndarray]:
        if i > 0:
            prev_rel = self.decompositions[f'relevant_{cell_type}'][layer][i - 1]
            prev_irrel = self.decompositions[f'irrelevant_{cell_type}'][layer][i - 1]
        else:
            if start < 0:
                prev_rel = self.activation_dict[layer, f'i{cell_type}x'][0, 0]
                prev_irrel = np.zeros(self.model.hidden_size, dtype=np.float32)
            else:
                prev_rel = np.zeros(self.model.hidden_size, dtype=np.float32)
                prev_irrel = self.activation_dict[layer, f'i{cell_type}x'][0, 0]

        return prev_rel, prev_irrel

    def _add_interactions(self, layer: int, i: int,
                          rel_gate: np.ndarray, rel_source: np.ndarray,
                          irrel_gate: np.ndarray, irrel_source: np.ndarray,
                          bias_gate: np.ndarray, bias_source: Optional[np.ndarray] = None,
                          cell_type: str = 'c',
                          inside_phrase: bool = False) -> None:
        """ Allows for interactions to be grouped differently than as specified in the paper. """
        rel_decomp_name = 'relevant_' + cell_type
        irrel_decomp_name = 'irrelevant_' + cell_type

        for decomp_name, interactions in ((rel_decomp_name, self.rel_interactions),
                                          (irrel_decomp_name, self.irrel_interactions)):
            for interaction in interactions:
                if interaction == 'rel-rel':
                    self.decompositions[decomp_name][layer][i] += rel_gate * rel_source
                elif interaction == 'rel-b':
                    self.decompositions[decomp_name][layer][i] += rel_source * bias_gate
                    if bias_source is not None:
                        self.decompositions[decomp_name][layer][i] += rel_gate * bias_source
                elif interaction == 'irrel-irrel':
                    self.decompositions[decomp_name][layer][i] += irrel_gate * irrel_source
                elif interaction == 'irrel-b':
                    if bias_source is not None:
                        self.decompositions[decomp_name][layer][i] += irrel_gate * bias_source
                    self.decompositions[decomp_name][layer][i] += irrel_source * bias_gate
                elif interaction == 'rel-irrel':
                    if decomp_name[:3] == 'rel' and self.only_source_rel:
                        self.decompositions[irrel_decomp_name][layer][i] += rel_gate * irrel_source
                        self.decompositions[decomp_name][layer][i] += rel_source * irrel_gate
                    else:
                        self.decompositions[decomp_name][layer][i] += rel_gate * irrel_source
                        self.decompositions[decomp_name][layer][i] += rel_source * irrel_gate
                elif interaction == 'b-b':
                    if bias_source is not None:
                        self._add_bias_bias(layer, i, bias_gate * bias_source, decomp_name,
                                            inside_phrase)
                else:
                    raise ValueError('Interaction type not understood')

    def _add_bias_bias(self, layer: int, i: int, bias_product: np.ndarray, decomp_name: str,
                       inside_phrase: bool) -> None:
        if decomp_name[:3] == 'rel':
            self.decompositions[decomp_name][layer][i] += bias_product
        else:
            if inside_phrase and self.bias_bias_only_in_phrase:
                self.decompositions['relevant_c'][layer][i] += bias_product
            else:
                self.decompositions['irrelevant_c'][layer][i] += bias_product

    def _reset_decompositions(self, start: int, slen: int, use_extracted_activations: bool) -> None:
        hidden_size = self.model.hidden_size
        num_layers = self.model.num_layers

        self.decompositions = {
            'relevant_c': np.zeros((num_layers, slen, hidden_size), dtype=np.float32),
            'relevant_h': np.zeros((num_layers, slen, hidden_size), dtype=np.float32),
            'irrelevant_c': np.zeros((num_layers, slen, hidden_size), dtype=np.float32),
            'irrelevant_h': np.zeros((num_layers, slen, hidden_size), dtype=np.float32),
        }

        # All indices until the start position of the relevant token won't yield any contribution,
        # so we can directly set the cell/hidden states that have been computed already.
        if not use_extracted_activations:
            return
        for l in range(num_layers):
            for i in range(start):
                self.decompositions['irrelevant_c'][l][i] = self.activation_dict[(l, 'cx')][0][i+1]
                self.decompositions['irrelevant_h'][l][i] = self.activation_dict[(l, 'hx')][0][i]

    def _set_rel_interactions(self, rel_interactions: Optional[List[str]]) -> None:
        all_interactions = {'rel-rel', 'rel-b', 'irrel-irrel', 'irrel-b', 'rel-irrel', 'b-b'}

        self.rel_interactions = set(rel_interactions or {'rel-rel', 'rel-irrel', 'rel-b'})
        self.irrel_interactions = all_interactions - self.rel_interactions
        assert not self.rel_interactions.intersection({'irrel-irrel', 'irrel-b'}), \
            'irrel-irrel and irrel-b can\'t be part of rel interactions'

    def _set_model_weights(self) -> None:
        for layer in range(self.model.num_layers):
            for name in ['ii', 'if', 'ig', 'io']:
                self.weight[layer, name] = self.model.weight[layer][name].detach().numpy()
                self.bias[layer, name[1]] = \
                    (self.model.bias[layer][f'i{name[1]}'].detach().numpy()
                     + self.model.bias[layer][f'h{name[1]}'].detach().numpy())
            for name in ['hi', 'hf', 'hg', 'ho']:
                self.weight[layer, name] = self.model.weight[layer][name].detach().numpy()

    def _validate_decomposition(self, scores: DecomposeArrayDict) -> None:
        final_hidden_state = self.get_final_activations((self.toplayer, 'hx'))
        original_score = np.ma.dot(final_hidden_state[0], self.decoder_w.T)

        decomposed_score = scores['relevant'][-1] + scores['irrelevant'][-1]

        # Sanity check: scores + irrel_scores should equal the LSTM's output minus bias
        assert np.allclose(original_score, decomposed_score, rtol=1e-3), \
            f'Decomposed score does not match: {original_score} vs {decomposed_score}'

    def _calc_scores(self) -> DecomposeArrayDict:
        return {
            'relevant': self.decompositions['relevant_h'][-1] @ self.decoder_w.T,
            'irrelevant': self.decompositions['irrelevant_h'][-1] @ self.decoder_w.T,
        }


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
