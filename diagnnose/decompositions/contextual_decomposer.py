import warnings
from typing import Any, Callable, List, Optional, Set, Tuple, Union

import torch
from overrides import overrides
from torch import Tensor

from diagnnose.typedefs.activations import (
    ActivationTensors,
    Decompositions,
    NamedTensors,
)

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

        self.weight: NamedTensors = {}
        self.bias: ActivationTensors = {}
        self.activations: NamedTensors = {}
        self.decompositions: Decompositions = {}

        self.rel_interactions: Set[str] = set()
        self.irrel_interactions: Set[str] = set()

    @overrides
    def _decompose(
        self,
        start: int,
        stop: int,
        rel_interactions: Optional[List[str]] = None,
        decompose_o: bool = False,
        bias_bias_only_in_phrase: bool = True,
        only_source_rel: bool = False,
        only_source_rel_b: bool = False,
        input_never_rel: bool = False,
        init_states_rel: bool = False,
        use_extracted_activations: bool = True,
        only_return_dec: bool = False,
    ) -> NamedTensors:
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
            rel-irrel. Defaults to rel-rel, rel-irrel & rel-b.
        bias_bias_only_in_phrase : bool, optional
            Toggles whether the bias-bias interaction should only be
            added when inside the relevant phrase. Defaults to True,
            indicating that only bias-bias interactions inside the
            subphrase range are added to the relevant decomposition.
        only_source_rel : bool, optional
            Relates to rel-irrel interactions. If set to true, only
            irrel_gate-rel_source interactions will be added to rel,
            similar to LRP (Arras et al., 2017).
        only_source_rel_b : bool, optional
            Relates to rel-b interactions. If set to true, only
            b-rel_source interactions will be added to rel,
            similar to LRP (Arras et al., 2017).
        input_never_rel : bool, optional
            Never add the Wx input to the rel part, useful when only
            investigating the model biases. Defaults to False.
        init_states_rel : bool, optional
            Directly add the initial cell/hidden states to the relevant
            part. Defaults to False.
        use_extracted_activations : bool, optional
            Allows previously extracted activations to be used to avoid
            unnecessary recomputations of those activations.
            Defaults to True.
        only_return_dec : bool, optional
            Only returns the decomposed cell states of the top layer,
            without calculating the corresponding decoder scores.
            Defaults to False.

        Returns
        -------
        scores : NamedArrayDict
            Dictionary with keys `relevant` and `irrelevant`, containing
            the decomposed scores for the earlier provided decoder.
        """
        self._set_rel_interactions(rel_interactions)

        start_index = max(0, start) if use_extracted_activations else 0
        slen = self.activation_dict[0, "emb"].shape[1]

        self._set_model_weights()
        self._reset_decompositions(start, slen, use_extracted_activations)

        self.bias_bias_only_in_phrase = bias_bias_only_in_phrase
        self.only_source_rel = only_source_rel
        self.only_source_rel_b = only_source_rel_b
        self.init_states_rel = init_states_rel

        for layer in range(self.model.num_layers):
            for i in range(start_index, slen):
                inside_phrase = start <= i < stop

                self._calc_activations(layer, i, start, inside_phrase, input_never_rel)

                self._add_forget_decomposition(layer, i, start)

                self._add_input_decomposition(layer, i, inside_phrase)

                self._add_output_decomposition(layer, i, decompose_o)

                self._project_hidden(layer, i)

        if only_return_dec:
            return self.decompositions[self.toplayer]

        scores = self._calc_scores()

        self._validate_decomposition(scores)

        return scores

    def _calc_activations(
        self, layer: int, i: int, start: int, inside_phrase: bool, input_never_rel: bool
    ) -> None:
        """ Recalculates the decomposed model activations.

        Input is either the word embedding in layer 0, or the beta/gamma
        decomposition of the hidden state in the previous layer.
        """
        if layer == 0:
            if inside_phrase and not input_never_rel:
                rel_input = self.activation_dict[0, "emb"][0][i]
                irrel_input = torch.zeros(
                    self.model.sizes[layer]["x"], dtype=torch.float32
                )
            else:
                rel_input = torch.zeros(
                    self.model.sizes[layer]["x"], dtype=torch.float32
                )
                irrel_input = self.activation_dict[0, "emb"][0][i]
        else:
            rel_input = self.decompositions[layer - 1]["rel_h"][i]
            irrel_input = self.decompositions[layer - 1]["irrel_h"][i]

        prev_rel_h, prev_irrel_h = self._get_prev_cells(layer, i, start, "h")

        if self.model.ih_concat_order == ["h", "i"]:
            rel_concat = torch.cat((prev_rel_h, rel_input), dim=1)
            irrel_concat = torch.cat((prev_irrel_h, irrel_input), dim=1)
        else:
            rel_concat = torch.cat((rel_input, prev_rel_h), dim=1)
            irrel_concat = torch.cat((irrel_input, prev_irrel_h), dim=1)

        # Weights are stored as 1 big array that project both input and hidden state.
        rel_proj = self.model.weight[layer] @ rel_concat
        irrel_proj = self.model.weight[layer] @ irrel_concat

        rel_names = map(lambda x: f"rel_{x}", self.model.split_order)
        self.activations.update(dict(zip(rel_names, torch.split(rel_proj, 4, dim=1))))

        irrel_names = map(lambda x: f"irrel_{x}", self.model.split_order)
        self.activations.update(
            dict(zip(irrel_names, torch.split(irrel_proj, 4, dim=1)))
        )

        if hasattr(self.model, "peepholes"):
            prev_rel_c, prev_irrel_c = self._get_prev_cells(layer, i, start, "c")
            for p in ["f", "i"]:
                self.activations[f"rel_{p}"] += (
                    prev_rel_c * self.model.peepholes[layer, p]
                )
                self.activations[f"irrel_{p}"] += (
                    prev_irrel_c * self.model.peepholes[layer, p]
                )

    def _add_forget_decomposition(self, layer: int, i: int, start: int) -> None:
        """ Calculates the forget gate decomposition, Equation (14) of the paper. """

        rel_contrib_f, irrel_contrib_f, bias_contrib_f = decomp_three(
            self.activations["rel_f"],
            self.activations["irrel_f"],
            self.bias[layer, "f"] + self.model.forget_offset,
            torch.sigmoid,
        )

        prev_rel_c, prev_irrel_c = self._get_prev_cells(layer, i, start, "c")

        self._add_interactions(
            layer,
            i,
            rel_contrib_f,
            prev_rel_c,
            irrel_contrib_f,
            prev_irrel_c,
            bias_contrib_f,
        )

    def _add_input_decomposition(self, layer: int, i: int, inside_phrase: bool) -> None:
        """ Calculates the input gate decomposition, Equation (17) of the paper. """

        rel_contrib_i, irrel_contrib_i, bias_contrib_i = decomp_three(
            self.activations["rel_i"],
            self.activations["irrel_i"],
            self.bias[layer, "i"],
            torch.sigmoid,
        )
        rel_contrib_g, irrel_contrib_g, bias_contrib_g = decomp_three(
            self.activations["rel_g"],
            self.activations["irrel_g"],
            self.bias[layer, "g"],
            torch.tanh,
        )

        self._add_interactions(
            layer,
            i,
            rel_contrib_i,
            rel_contrib_g,
            irrel_contrib_i,
            irrel_contrib_g,
            bias_contrib_i,
            bias_source=bias_contrib_g,
            inside_phrase=inside_phrase,
        )

    def _add_output_decomposition(self, layer: int, i: int, decompose_o: bool) -> None:
        """ Calculates the output gate decomposition, Equation (23).

        As stated in the paper, output decomposition is not always
        beneficial and can therefore be toggled off.
        """
        rel_c = self.decompositions[layer]["rel_c"][i]
        irrel_c = self.decompositions[layer]["irrel_c"][i]

        new_rel_h, new_irrel_h = decomp_tanh_two(rel_c, irrel_c)
        rel_o, irrel_o = self.activations["rel_o"], self.activations["irrel_o"]

        if hasattr(self.model, "peepholes"):
            rel_o += rel_c * self.model.peepholes[layer, "o"]
            irrel_o += irrel_c * self.model.peepholes[layer, "o"]

        if decompose_o:
            rel_contrib_o, irrel_contrib_o, bias_contrib_o = decomp_three(
                rel_o, irrel_o, self.bias[layer, "o"], torch.sigmoid
            )

            self._add_interactions(
                layer,
                i,
                rel_contrib_o,
                new_rel_h,
                irrel_contrib_o,
                new_irrel_h,
                bias_contrib_o,
                cell_type="h_wo_proj",
            )
        else:
            o = torch.sigmoid(rel_o + irrel_o + self.bias[layer, "o"])
            new_rel_h *= o
            new_irrel_h *= o

            self.decompositions[layer]["rel_h_wo_proj"][i] = new_rel_h
            self.decompositions[layer]["irrel_h_wo_proj"][i] = new_irrel_h

    def _project_hidden(self, layer: int, i: int) -> None:
        if self.model.sizes[layer]["h"] != self.model.sizes[layer]["c"]:
            c2h_wo_proj = self.model.lstm.weight_P[layer]

            self.decompositions[layer]["rel_h"][i] = (
                self.decompositions[layer]["rel_h_wo_proj"][i] @ c2h_wo_proj
            )

            self.decompositions[layer]["irrel_h"][i] = (
                self.decompositions[layer]["irrel_h_wo_proj"][i] @ c2h_wo_proj
            )

        else:
            self.decompositions[layer]["rel_h"][i] = self.decompositions[layer][
                "rel_h_wo_proj"
            ][i]

            self.decompositions[layer]["irrel_h"][i] = self.decompositions[layer][
                "irrel_h_wo_proj"
            ][i]

    def _get_prev_cells(
        self, layer: int, i: int, start: int, cell_type: str
    ) -> Tuple[Tensor, Tensor]:
        hidden_size = self.model.sizes[layer][cell_type]

        if i > 0:
            prev_rel = self.decompositions[layer][f"rel_{cell_type}"][i - 1]
            prev_irrel = self.decompositions[layer][f"irrel_{cell_type}"][i - 1]
        else:
            if start < 0 or self.init_states_rel:
                prev_rel = self.activation_dict[layer, f"i{cell_type}x"][0, 0]
                prev_irrel = torch.zeros(hidden_size, dtype=torch.float32)
            else:
                prev_rel = torch.zeros(hidden_size, dtype=torch.float32)
                prev_irrel = self.activation_dict[layer, f"i{cell_type}x"][0, 0]

        return prev_rel, prev_irrel

    def _add_interactions(
        self,
        layer: int,
        i: int,
        rel_gate: Tensor,
        rel_source: Tensor,
        irrel_gate: Tensor,
        irrel_source: Tensor,
        bias_gate: Tensor,
        bias_source: Optional[Tensor] = None,
        cell_type: str = "c",
        inside_phrase: bool = False,
    ) -> None:
        """ Allows for interactions to be grouped differently than as specified in the paper. """
        rel_cd_name = f"rel_{cell_type}"
        irrel_cd_name = f"irrel_{cell_type}"

        for decomp_name, interactions in (
            (rel_cd_name, self.rel_interactions),
            (irrel_cd_name, self.irrel_interactions),
        ):
            for interaction in interactions:
                if interaction == "rel-rel":
                    self.decompositions[layer][decomp_name][i] += rel_gate * rel_source

                elif interaction == "rel-b":
                    self.decompositions[layer][decomp_name][i] += bias_gate * rel_source
                    if bias_source is not None:
                        if self.only_source_rel_b:
                            self.decompositions[layer][irrel_cd_name][i] += (
                                rel_gate * bias_source
                            )
                        else:
                            self.decompositions[layer][decomp_name][i] += (
                                rel_gate * bias_source
                            )

                elif interaction == "irrel-irrel":
                    self.decompositions[layer][decomp_name][i] += (
                        irrel_gate * irrel_source
                    )

                elif interaction == "irrel-b":
                    if bias_source is not None:
                        self.decompositions[layer][decomp_name][i] += (
                            irrel_gate * bias_source
                        )
                    self.decompositions[layer][decomp_name][i] += (
                        bias_gate * irrel_source
                    )

                elif interaction == "rel-irrel":
                    if decomp_name.startswith("rel") and self.only_source_rel:
                        self.decompositions[layer][irrel_cd_name][i] += (
                            rel_gate * irrel_source
                        )
                        self.decompositions[layer][decomp_name][i] += (
                            irrel_gate * rel_source
                        )
                    else:
                        self.decompositions[layer][decomp_name][i] += (
                            rel_gate * irrel_source
                        )
                        self.decompositions[layer][decomp_name][i] += (
                            irrel_gate * rel_source
                        )

                elif interaction == "b-b":
                    if bias_source is not None:
                        self._add_bias_bias(
                            layer,
                            i,
                            bias_gate * bias_source,
                            decomp_name,
                            inside_phrase,
                        )

                else:
                    raise ValueError(f"Interaction type not understood: {interaction}")

    def _add_bias_bias(
        self,
        layer: int,
        i: int,
        bias_product: Tensor,
        decomp_name: str,
        inside_phrase: bool,
    ) -> None:
        if decomp_name.startswith("rel"):
            self.decompositions[layer][decomp_name][i] += bias_product
        else:
            if inside_phrase and self.bias_bias_only_in_phrase:
                self.decompositions[layer]["rel_c"][i] += bias_product
            else:
                self.decompositions[layer]["irrel_c"][i] += bias_product

    def _reset_decompositions(
        self, start: int, slen: int, use_extracted_activations: bool
    ) -> None:
        num_layers = self.model.num_layers

        # The h_wo_proj decompositions are used for lstms with differing hidden and cell sizes.
        # The h decomposition is then created by the projection of `h_wo_proj` in `_project_hidden`.
        self.decompositions = {
            layer: {
                "rel_c": torch.zeros(
                    (slen, self.model.sizes[layer]["c"]), dtype=torch.float32
                ),
                "rel_h": torch.zeros(
                    (slen, self.model.sizes[layer]["h"]), dtype=torch.float32
                ),
                "rel_h_wo_proj": torch.zeros(
                    (slen, self.model.sizes[layer]["c"]), dtype=torch.float32
                ),
                "irrel_c": torch.zeros(
                    (slen, self.model.sizes[layer]["c"]), dtype=torch.float32
                ),
                "irrel_h": torch.zeros(
                    (slen, self.model.sizes[layer]["h"]), dtype=torch.float32
                ),
                "irrel_h_wo_proj": torch.zeros(
                    (slen, self.model.sizes[layer]["c"]), dtype=torch.float32
                ),
            }
            for layer in range(self.model.num_layers)
        }

        # All indices until the start position of the relevant token won't yield any contribution,
        # so we can directly set the cell/hidden states that have been computed already.
        if use_extracted_activations:
            for l in range(num_layers):
                for i in range(start):
                    self.decompositions[l]["irrel_c"][i] = self.activation_dict[
                        (l, "cx")
                    ][0][i + 1]
                    self.decompositions[l]["irrel_h"][i] = self.activation_dict[
                        (l, "hx")
                    ][0][i + 1]

    def _set_rel_interactions(self, rel_interactions: Optional[List[str]]) -> None:
        all_interactions = {
            "rel-rel",
            "rel-b",
            "irrel-irrel",
            "irrel-b",
            "rel-irrel",
            "b-b",
        }

        self.rel_interactions = set(
            rel_interactions or {"rel-rel", "rel-irrel", "rel-b"}
        )
        self.irrel_interactions = all_interactions - self.rel_interactions
        assert not self.rel_interactions.intersection(
            {"irrel-irrel", "irrel-b"}
        ), "irrel-irrel and irrel-b can't be part of rel interactions"

    def _set_model_weights(self) -> None:
        for layer in range(self.model.num_layers):
            bias = self.model.bias[layer]

            if isinstance(self.model.weight[layer], torch.Tensor):
                self.model.weight[layer] = self.model.weight[layer].detach().numpy()

            self.bias.update(
                {
                    (layer, name): tensor
                    for name, tensor in zip(
                        self.model.split_order, torch.split(bias, 4, dim=1)
                    )
                }
            )

    def _validate_decomposition(self, scores: NamedTensors) -> None:
        final_hidden_state = self.get_final_activations((self.toplayer, "hx"))
        original_score = final_hidden_state[0] @ self.decoder_w.t()

        decomposed_score = scores["relevant"][-1] + scores["irrelevant"][-1]

        avg_difference = torch.mean(original_score - decomposed_score)
        max_difference = torch.max(torch.abs(original_score - decomposed_score))

        # Sanity check: scores + irrel_scores should equal the original output
        if not torch.allclose(original_score, decomposed_score, rtol=1e-3):
            warnings.warn(
                f"Decomposed scores do not match: orig {original_score} vs dec {decomposed_score}\n"
                f"Average difference: {avg_difference}\n"
                f"Maximum difference: {max_difference}\n"
                f"If the difference is small (<<1e-3) this is likely due to numerical instability "
                f"and hence not necessarily problematic."
            )

    def _calc_scores(self) -> NamedTensors:
        return {
            "relevant": self.decompositions[self.toplayer]["rel_h"] @ self.decoder_w.T,
            "irrelevant": self.decompositions[self.toplayer]["irrel_h"]
            @ self.decoder_w.T,
        }


# Activation linearizations as described in chapter 3.2.2
def decomp_three(
    a: Tensor, b: Tensor, c: Tensor, activation: Callable[[Tensor], Tensor]
) -> Tuple[Tensor, Tensor, Tensor]:
    ac = activation(a + c)
    bc = activation(b + c)
    abc = activation(a + b + c)

    c_contrib = activation(c)
    a_contrib = 0.5 * ((ac - c_contrib) + (abc - bc))
    b_contrib = 0.5 * ((bc - c_contrib) + (abc - ac))

    return a_contrib, b_contrib, c_contrib


def decomp_tanh_two(a: Tensor, b: Tensor) -> Tuple[Tensor, Tensor]:
    atanh = torch.tanh(a)
    btanh = torch.tanh(b)
    abtanh = torch.tanh(a + b)

    return 0.5 * (atanh + (abtanh - btanh)), 0.5 * (btanh + (abtanh - atanh))
