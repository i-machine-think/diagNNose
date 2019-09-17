import warnings
from typing import Any, Callable, List, Optional, Set, Tuple, Union

import torch
from overrides import overrides
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence

import diagnnose.typedefs.config as config
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.weight: NamedTensors = {}
        self.bias: ActivationTensors = {}
        self.activations: NamedTensors = {}
        self.decompositions: Decompositions = {}

        self.rel_interactions: Set[str] = set()
        self.irrel_interactions: Set[str] = set()

    @overrides
    def _decompose(
        self,
        start: Union[int, Tensor],
        stop: Union[int, Tensor],
        rel_interactions: Optional[List[str]] = None,
        fix_shapley: bool = True,
        decompose_o: bool = False,
        bias_bias_only_in_phrase: bool = True,
        only_source_rel: bool = False,
        only_source_rel_b: bool = False,
        input_never_rel: bool = False,
        init_states_rel: bool = False,
        use_extracted_activations: bool = True,
    ) -> NamedTensors:
        """ Main loop for the contextual decomposition.

        Parameters
        ----------
        start : int | Tensor
            Starting index of the relevant subphrase. Pass a Tensor
            to pass along different indices for batch elements.
        stop : int | Tensor
            Stopping index of the relevant subphrase. This stop index
            is not included in the subphrase range, similar to range().
            Pass a Tensor to pass along different indices for batch
            elements.
        decompose_o : bool, optional
            Toggles decomposition of the output gate. Defaults to False.
        rel_interactions : List[str], optional
            Indicates the interactions that are part of the relevant
            decomposition. Possible interactions are: rel-rel, rel-b and
            rel-irrel. Defaults to rel-rel, rel-irrel & rel-b.
        fix_shapley : bool, optional
            Fix the bias term position in the gate linearization, as
            is done in the original paper. Setting this to True will 
            lead to a very considerable _bias_ towards the bias!
            Defaults to True.
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

        Returns
        -------
        scores : NamedArrayDict
            Dictionary with keys `relevant` and `irrelevant`, containing
            the decomposed scores for the earlier provided decoder.
        """
        self._set_rel_interactions(rel_interactions)
        if fix_shapley:
            self.linearize_gate = shapley_three_fixed
        else:
            self.linearize_gate = shapley_three

        if isinstance(start, int) and use_extracted_activations:
            start_index = max(0, start)
        else:
            start_index = 0
        batch_size, slen = self.activation_dict[0, "emb"].shape[:2]

        self._split_model_bias()
        self._reset_decompositions(start, batch_size, slen, use_extracted_activations)

        self.bias_bias_only_in_phrase = bias_bias_only_in_phrase
        self.only_source_rel = only_source_rel
        self.only_source_rel_b = only_source_rel_b
        self.init_states_rel = init_states_rel

        for layer in range(self.model.num_layers):
            for i in range(start_index, slen):
                inside_phrase = self._get_inside_phrase(start, stop, i)

                self._calc_activations(layer, i, start, inside_phrase, input_never_rel)

                self._add_forget_decomposition(layer, i, start)

                self._add_input_decomposition(layer, i, inside_phrase)

                self._add_output_decomposition(layer, i, decompose_o)

                self._project_hidden(layer, i)

        self._validate_decomposition()

        return self._calc_scores()

    @staticmethod
    def _get_inside_phrase(start: Union[int, Tensor], stop: Union[int, Tensor], i: int):
        """ Returns bool or bool tensor indicating whether current
        position i is part of the relevant phrase.
        """
        if isinstance(start, int) and isinstance(stop, int):
            return start <= i < stop
        else:
            inside_phrase = (start <= i) * (i < stop)
            return inside_phrase.unsqueeze(1)

    def _calc_activations(
        self,
        layer: int,
        i: int,
        start: Union[int, Tensor],
        inside_phrase: Union[bool, Tensor],
        input_never_rel: bool,
    ) -> None:
        """ Recalculates the decomposed model activations.

        Input is either the word embedding in layer 0, or the beta/gamma
        decomposition of the hidden state in the previous layer.
        """
        if layer == 0:
            if isinstance(inside_phrase, Tensor) and not input_never_rel:
                # inside_phrase now acts as a mask on the word embeddings
                pos_inside_phrase = inside_phrase.to(config.DTYPE)
                neg_inside_phrase = (~inside_phrase).to(config.DTYPE)
                rel_input = self.activation_dict[0, "emb"][:, i] * pos_inside_phrase
                irrel_input = self.activation_dict[0, "emb"][:, i] * neg_inside_phrase
            elif inside_phrase and not input_never_rel:
                rel_input = self.activation_dict[0, "emb"][:, i]
                irrel_input = torch.zeros(rel_input.shape, dtype=config.DTYPE)
            else:
                irrel_input = self.activation_dict[0, "emb"][:, i]
                rel_input = torch.zeros(irrel_input.shape, dtype=config.DTYPE)
        else:
            rel_input = self.decompositions[layer - 1]["rel_h"][:, i]
            irrel_input = self.decompositions[layer - 1]["irrel_h"][:, i]

        prev_rel_h, prev_irrel_h = self._get_prev_cells(layer, i, start, "h")

        if self.model.ih_concat_order == ["h", "i"]:
            rel_concat = torch.cat((prev_rel_h, rel_input), dim=1)
            irrel_concat = torch.cat((prev_irrel_h, irrel_input), dim=1)
        else:
            rel_concat = torch.cat((rel_input, prev_rel_h), dim=1)
            irrel_concat = torch.cat((irrel_input, prev_irrel_h), dim=1)

        # Weights are stored as 1 big array that project both input and hidden state.
        rel_proj = rel_concat @ self.model.weight[layer]
        irrel_proj = irrel_concat @ self.model.weight[layer]

        # Split weight projection back to the individual gates.
        rel_names = map(lambda x: f"rel_{x}", self.model.split_order)
        self.activations.update(dict(zip(rel_names, torch.chunk(rel_proj, 4, dim=1))))

        irrel_names = map(lambda x: f"irrel_{x}", self.model.split_order)
        self.activations.update(
            dict(zip(irrel_names, torch.chunk(irrel_proj, 4, dim=1)))
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

        rel_contrib_f, irrel_contrib_f, bias_contrib_f = self.linearize_gate(
            [
                self.activations["rel_f"],
                self.activations["irrel_f"],
                self.bias[layer, "f"] + self.model.forget_offset,
            ],
            gate=torch.sigmoid,
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

    def _add_input_decomposition(
        self, layer: int, i: int, inside_phrase: Union[bool, Tensor]
    ) -> None:
        """ Calculates the input gate decomposition, Equation (17) of the paper. """

        rel_contrib_i, irrel_contrib_i, bias_contrib_i = self.linearize_gate(
            [
                self.activations["rel_i"],
                self.activations["irrel_i"],
                self.bias[layer, "i"],
            ],
            gate=torch.sigmoid,
        )
        self.rel_contrib_i = rel_contrib_i
        self.bias_contrib_i = bias_contrib_i[0]
        rel_contrib_g, irrel_contrib_g, bias_contrib_g = self.linearize_gate(
            [
                self.activations["rel_g"],
                self.activations["irrel_g"],
                self.bias[layer, "g"],
            ]
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
        rel_c = self.decompositions[layer]["rel_c"][:, i]
        irrel_c = self.decompositions[layer]["irrel_c"][:, i]

        new_rel_h, new_irrel_h = shapley_two([rel_c, irrel_c])
        rel_o, irrel_o = self.activations["rel_o"], self.activations["irrel_o"]

        if hasattr(self.model, "peepholes"):
            rel_o += rel_c * self.model.peepholes[layer, "o"]
            irrel_o += irrel_c * self.model.peepholes[layer, "o"]

        if decompose_o:
            rel_contrib_o, irrel_contrib_o, bias_contrib_o = self.linearize_gate(
                [rel_o, irrel_o, self.bias[layer, "o"]], gate=torch.sigmoid
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

            self.decompositions[layer]["rel_h_wo_proj"][:, i] = new_rel_h
            self.decompositions[layer]["irrel_h_wo_proj"][:, i] = new_irrel_h

    def _project_hidden(self, layer: int, i: int) -> None:
        if self.model.sizes[layer]["h"] != self.model.sizes[layer]["c"]:
            c2h_wo_proj = self.model.lstm.weight_P[layer]

            self.decompositions[layer]["rel_h"][:, i] = (
                self.decompositions[layer]["rel_h_wo_proj"][:, i] @ c2h_wo_proj
            )

            self.decompositions[layer]["irrel_h"][:, i] = (
                self.decompositions[layer]["irrel_h_wo_proj"][:, i] @ c2h_wo_proj
            )

        else:
            self.decompositions[layer]["rel_h"][:, i] = self.decompositions[layer][
                "rel_h_wo_proj"
            ][:, i]

            self.decompositions[layer]["irrel_h"][:, i] = self.decompositions[layer][
                "irrel_h_wo_proj"
            ][:, i]

    def _get_prev_cells(
        self, layer: int, i: int, start: Union[int, Tensor], cell_type: str
    ) -> Tuple[Tensor, Tensor]:
        if i > 0:
            prev_rel = self.decompositions[layer][f"rel_{cell_type}"][:, i - 1]
            prev_irrel = self.decompositions[layer][f"irrel_{cell_type}"][:, i - 1]
        else:
            init_rel = start < 0
            if isinstance(start, Tensor) and not self.init_states_rel:
                pos_init_rel = init_rel.unsqueeze(1).to(config.DTYPE)
                neg_init_rel = (~init_rel.unsqueeze(1)).to(config.DTYPE)
                prev_rel = self.activation_dict[layer, f"i{cell_type}x"] * pos_init_rel
                prev_irrel = (
                    self.activation_dict[layer, f"i{cell_type}x"] * neg_init_rel
                )
            elif self.init_states_rel or init_rel:
                prev_rel = self.activation_dict[layer, f"i{cell_type}x"]
                prev_irrel = torch.zeros(prev_rel.shape, dtype=config.DTYPE)
            else:
                prev_irrel = self.activation_dict[layer, f"i{cell_type}x"]
                prev_rel = torch.zeros(prev_irrel.shape, dtype=config.DTYPE)

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
        inside_phrase: Union[bool, Tensor] = False,
    ) -> None:
        """ Allows for interactions to be grouped differently than as specified in the paper. """
        rel_cd_name = f"rel_{cell_type}"
        irrel_cd_name = f"irrel_{cell_type}"

        for dec_name, interactions in (
            (rel_cd_name, self.rel_interactions),
            (irrel_cd_name, self.irrel_interactions),
        ):
            for interaction in interactions:
                if interaction == "rel-rel":
                    self.decompositions[layer][dec_name][:, i] += rel_gate * rel_source

                elif interaction == "rel-b":
                    self.decompositions[layer][dec_name][:, i] += bias_gate * rel_source
                    if bias_source is not None:
                        if self.only_source_rel_b:
                            self.decompositions[layer][irrel_cd_name][:, i] += (
                                rel_gate * bias_source
                            )
                        else:
                            self.decompositions[layer][dec_name][:, i] += (
                                rel_gate * bias_source
                            )

                elif interaction == "irrel-irrel":
                    self.decompositions[layer][dec_name][:, i] += (
                        irrel_gate * irrel_source
                    )

                elif interaction == "irrel-b":
                    if bias_source is not None:
                        self.decompositions[layer][dec_name][:, i] += (
                            irrel_gate * bias_source
                        )
                    self.decompositions[layer][dec_name][:, i] += (
                        bias_gate * irrel_source
                    )

                elif interaction == "rel-irrel":
                    if dec_name.startswith("rel") and self.only_source_rel:
                        self.decompositions[layer][irrel_cd_name][:, i] += (
                            rel_gate * irrel_source
                        )
                        self.decompositions[layer][dec_name][:, i] += (
                            irrel_gate * rel_source
                        )
                    else:
                        self.decompositions[layer][dec_name][:, i] += (
                            rel_gate * irrel_source
                        )
                        self.decompositions[layer][dec_name][:, i] += (
                            irrel_gate * rel_source
                        )

                elif interaction == "b-b":
                    if bias_source is not None:
                        self._add_bias_bias(
                            layer, i, bias_gate * bias_source, dec_name, inside_phrase
                        )

                else:
                    raise ValueError(f"Interaction type not understood: {interaction}")

    def _add_bias_bias(
        self,
        layer: int,
        i: int,
        bias_product: Tensor,
        dec_name: str,
        inside_phrase: Union[bool, Tensor],
    ) -> None:
        if dec_name.startswith("rel"):
            self.decompositions[layer][dec_name][:, i] += bias_product
        else:
            if self.bias_bias_only_in_phrase and isinstance(inside_phrase, Tensor):
                pos_inside_phrase = inside_phrase.to(config.DTYPE)
                neg_inside_phrase = (~inside_phrase).to(config.DTYPE)
                self.decompositions[layer]["rel_c"][:, i] += (
                    bias_product * pos_inside_phrase
                )
                self.decompositions[layer]["irrel_c"][:, i] += (
                    bias_product * neg_inside_phrase
                )
            elif self.bias_bias_only_in_phrase and inside_phrase:
                self.decompositions[layer]["rel_c"][:, i] += bias_product
            else:
                self.decompositions[layer]["irrel_c"][:, i] += bias_product

    def _reset_decompositions(
        self, start: int, batch_size: int, slen: int, use_extracted_activations: bool
    ) -> None:
        num_layers = self.model.num_layers
        sizes = self.model.sizes

        # The h_wo_proj decompositions are used for lstms with differing hidden and cell sizes.
        # The h decomposition is then created by the projection of `h_wo_proj` in `_project_hidden`.
        self.decompositions = {
            layer: {
                "rel_c": torch.zeros(
                    (batch_size, slen, sizes[layer]["c"]), dtype=config.DTYPE
                ),
                "rel_h": torch.zeros(
                    (batch_size, slen, sizes[layer]["h"]), dtype=config.DTYPE
                ),
                "rel_h_wo_proj": torch.zeros(
                    (batch_size, slen, sizes[layer]["c"]), dtype=config.DTYPE
                ),
                "irrel_c": torch.zeros(
                    (batch_size, slen, sizes[layer]["c"]), dtype=config.DTYPE
                ),
                "irrel_h": torch.zeros(
                    (batch_size, slen, sizes[layer]["h"]), dtype=config.DTYPE
                ),
                "irrel_h_wo_proj": torch.zeros(
                    (batch_size, slen, sizes[layer]["c"]), dtype=config.DTYPE
                ),
            }
            for layer in range(num_layers)
        }

        # All indices until the start position of the relevant token won't yield any contribution,
        # so we can directly set the cell/hidden states that have been computed already.
        if use_extracted_activations and isinstance(start, int):
            for l in range(num_layers):
                for i in range(start):
                    self.decompositions[l]["irrel_c"][:, i] = self.activation_dict[
                        (l, "cx")
                    ][:, i + 1]
                    self.decompositions[l]["irrel_h"][:, i] = self.activation_dict[
                        (l, "hx")
                    ][:, i + 1]

    def _set_rel_interactions(self, rel_interactions: Optional[List[str]]) -> None:
        all_interactions = {
            "rel-rel",
            "rel-b",
            "irrel-irrel",
            "irrel-b",
            "rel-irrel",
            "b-b",
        }

        self.rel_interactions = set(rel_interactions or {"rel-rel", "rel-b"})
        self.irrel_interactions = all_interactions - self.rel_interactions
        assert not self.rel_interactions.intersection(
            {"irrel-irrel", "irrel-b"}
        ), "irrel-irrel and irrel-b can't be part of rel interactions"

    def _split_model_bias(self) -> None:
        for layer in range(self.model.num_layers):
            bias = self.model.bias[layer]

            self.bias.update(
                {
                    (layer, name): tensor
                    for name, tensor in zip(
                        self.model.split_order, torch.chunk(bias, 4, dim=0)
                    )
                }
            )

    def _validate_decomposition(self) -> None:
        true_hidden = self.activation_dict[self.toplayer, "hx"][:, 1:]
        true_hidden = pack_padded_sequence(
            true_hidden,
            lengths=self.final_index,
            batch_first=True,
            enforce_sorted=False,
        ).data

        dec_hidden = (
            self.decompositions[self.toplayer]["rel_h"]
            + self.decompositions[self.toplayer]["irrel_h"]
        )
        dec_hidden = pack_padded_sequence(
            dec_hidden, lengths=self.final_index, batch_first=True, enforce_sorted=False
        ).data

        avg_difference = torch.mean(true_hidden - dec_hidden)
        max_difference = torch.max(torch.abs(true_hidden - dec_hidden))

        assert torch.sum(torch.isnan(avg_difference)) == 0

        # Sanity check: scores + irrel_scores should equal the original output
        if (
            not torch.allclose(true_hidden, dec_hidden, rtol=1e-3)
            and max_difference > 1e-3
        ):
            warnings.warn(
                f"Decomposed scores don't match: orig {true_hidden} vs dec {dec_hidden}\n"
                f"Average difference: {avg_difference}\n"
                f"Maximum difference: {max_difference}\n"
                f"If the difference is small (<<1e-3) this is likely due to numerical instability "
                f"and hence not necessarily problematic."
            )

    def _calc_scores(self) -> NamedTensors:
        rel_dec = self.decompositions[self.toplayer]["rel_h"]
        irrel_dec = self.decompositions[self.toplayer]["irrel_h"]

        if self.extra_classes is not None:
            for i, j in enumerate(self.extra_classes, start=1):
                rel_dec[:, -i] = rel_dec[:, j]
                irrel_dec[:, -i] = irrel_dec[:, j]

        # If no classes have been provided the decomposed states themselves are returned
        if self.decoder_b.size(1) == 0:
            decomp_dict = {"relevant": rel_dec, "irrelevant": irrel_dec}

            return decomp_dict

        score_dict = {
            "relevant": torch.bmm(rel_dec, self.decoder_w),
            "irrelevant": torch.bmm(irrel_dec, self.decoder_w),
        }

        return score_dict


# Activation linearizations as described in chapter 3.2.2
def shapley_three(
    tensors: List[Tensor], gate: Callable[[Tensor], Tensor] = torch.tanh
) -> List[Tensor]:
    a, b, c = tensors
    ac = gate(a + c)
    ab = gate(a + b)
    bc = gate(b + c)
    abc = gate(a + b + c)
    a = gate(a)
    b = gate(b)
    c = gate(c)

    a_contrib = (1 / 6) * (2 * (abc - bc) + (ab - b) + (ac - c) + 2 * a)
    b_contrib = (1 / 6) * (2 * (abc - ac) + (ab - a) + (bc - c) + 2 * b)
    c_contrib = (1 / 6) * (2 * (abc - ab) + (bc - b) + (ac - a) + 2 * c)

    return [a_contrib, b_contrib, c_contrib]


def shapley_two(
    tensors: List[Tensor], gate: Callable[[Tensor], Tensor] = torch.tanh
) -> List[Tensor]:
    a, b = tensors
    ab = gate(a + b)
    a = gate(a)
    b = gate(b)

    a_contrib = 0.5 * (a + (ab - b))
    b_contrib = 0.5 * (b + (ab - a))

    return [a_contrib, b_contrib]


# The original implementation sets the bias term fixed
# to the first position in the Shapley approximation.
def shapley_three_fixed(
    tensors: List[Tensor], gate: Callable[[Tensor], Tensor] = torch.tanh
) -> List[Tensor]:
    a, b, c = tensors
    ac = gate(a + c)
    bc = gate(b + c)
    abc = gate(a + b + c)

    c_contrib = gate(c)

    a_contrib = (1 / 2) * ((abc - bc) + (ac - c_contrib))
    b_contrib = (1 / 2) * ((abc - ac) + (bc - c_contrib))

    return [a_contrib, b_contrib, c_contrib]
