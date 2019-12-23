import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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

from .contextual_decomposer import BaseDecomposer
from .shapley_full import calc_shapley_values


InputPartition = Union[Tuple[int, int], List[int]]


class ShapleyDecomposer(BaseDecomposer):
    """ Shapley Decomposer, uses the mechanism of CD but considers all
    input features simultaneously.

    Inherits and uses functions from BaseDecomposer.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.weight: NamedTensors = {}
        self.bias: ActivationTensors = {}
        self.activations: NamedTensors = {}
        self.decompositions: Decompositions = {}

        self.rel_interactions: Set[str] = set()
        self.idx2partition_idx: Dict[int, int] = {}
        self.num_partitions: int = 0

    @overrides
    def _decompose(
        self,
        input_partitions: Optional[List[InputPartition]] = None,
        rel_interactions: Optional[List[str]] = None,
        decompose_o: bool = False,
        use_extracted_activations: bool = True,
    ) -> Tensor:
        """ Main loop for the contextual decomposition.

        Parameters
        ----------
        input_partitions : List[InputPartition], optional
            Optional list of input partitions: represented as either
            a tuple of (start, stop) spans, or lists of indices that
            should be grouped together. If not provided each individual
            input will stand on its own.
        rel_interactions : List[str], optional
            Indicates the interactions that are part of the relevant
            decomposition. Possible interactions are: rel-rel, rel-b and
            rel-irrel. Defaults to rel-rel, rel-irrel & rel-b.
        decompose_o : bool, optional
            Toggles decomposition of the output gate. Defaults to False.
        use_extracted_activations : bool, optional
            Allows previously extracted activations to be used to avoid
            unnecessary recomputations of those activations.
            Defaults to True. TODO: check if worthwhile to add

        Returns
        -------
        scores : NamedArrayDict
            Dictionary with keys `relevant` and `irrelevant`, containing
            the decomposed scores for the earlier provided decoder.
        """
        self._split_model_bias(self.batch_size)
        self._set_input_partitions(input_partitions, self.slen)
        self._reset_decompositions(self.batch_size, self.slen)

        for layer in range(self.model.num_layers):
            for i in range(1, self.slen + 1):
                self._calc_activations(layer, i)

                self._add_forget_decomposition(layer, i)

                self._add_input_decomposition(layer, i)

                self._add_output_decomposition(layer, i, decompose_o)

                self._project_hidden(layer, i)

        self._validate_decomposition()

        return self._calc_scores()

    def _calc_activations(self, layer: int, i: int) -> None:
        """ Recalculates the decomposed model activations.

        Input is either the word embedding in layer 0, or the beta/gamma
        decomposition of the hidden state in the previous layer.
        """
        if layer == 0:
            # We index embs with i-1 because there are no embedding activations for the 0th step,
            # which is set to be the index for the initial states.
            # (batch_size, num_partitions)
            embs = self.activation_dict[0, "emb"][:, i - 1]
            nhid = embs.size(1)
            input_ = torch.zeros(
                (self.batch_size, self.num_partitions, nhid), dtype=config.DTYPE
            )
            p_idx = self.idx2partition_idx[i]
            input_[:, p_idx] = embs
        else:
            input_ = self.decompositions[layer - 1]["h"][:, :, i]

        prev_h = self.decompositions[layer]["h"][:, :, i - 1]

        if self.model.ih_concat_order == ["i", "h"]:
            ih_concat = torch.cat((input_, prev_h), dim=2)
        else:
            ih_concat = torch.cat((prev_h, input_), dim=2)

        # Weights are stored as 1 big array that project both input and hidden state.
        ih_proj = ih_concat @ self.model.weight[layer]

        # Split weight projection back to the individual gates.
        self.activations.update(
            dict(zip(self.model.split_order, torch.chunk(ih_proj, 4, dim=2)))
        )

        if hasattr(self.model, "peepholes"):
            prev_c = self.decompositions[layer]["c"][:, :, i - 1]
            for p in ["f", "i"]:
                self.activations[p] += prev_c * self.model.peepholes[layer, p]

    def _add_forget_decomposition(self, layer: int, i: int) -> None:
        """ Calculates the forget gate decomposition, Equation (14) of the paper. """
        shapley_f = calc_shapley_values(
            torch.cat((self.activations["f"], self.bias[layer, "f"]), dim=1),
            torch.sigmoid,
        )

        prev_c = self.decompositions[layer]["c"][:, :, i - 1]

        self._add_interactions(layer, i, shapley_f, prev_c)

    def _add_input_decomposition(self, layer: int, i: int) -> None:
        """ Calculates the input gate decomposition, Equation (17) of the paper. """
        shapley_i = calc_shapley_values(
            torch.cat((self.activations["i"], self.bias[layer, "i"]), dim=1),
            torch.sigmoid,
        )

        shapley_g = calc_shapley_values(
            torch.cat((self.activations["g"], self.bias[layer, "g"]), dim=1), torch.tanh
        )

        self._add_interactions(layer, i, shapley_i, shapley_g)

    def _add_output_decomposition(self, layer: int, i: int, decompose_o: bool) -> None:
        """ Calculates the output gate decomposition, Equation (23).

        As stated in the paper, output decomposition is not always
        beneficial and can therefore be toggled off.
        """
        c = self.decompositions[layer]["c"][:, :, i]

        shapley_c = calc_shapley_values(c, torch.tanh)

        if hasattr(self.model, "peepholes"):
            self.activations["o"] += c * self.model.peepholes[layer, "o"]

        if decompose_o:
            shapley_o = calc_shapley_values(
                torch.cat((self.activations["o"], self.bias[layer, "o"]), dim=1),
                torch.sigmoid,
            )

            self._add_interactions(
                layer, i, shapley_o, shapley_c, cell_type="h_wo_proj"
            )
        else:
            o = torch.sigmoid(self.activations["o"].sum(1) + self.bias[layer, "o"])
            h_wo_proj = o * shapley_c

            self.decompositions[layer]["h_wo_proj"][:, :, i] = h_wo_proj

    def _project_hidden(self, layer: int, i: int) -> None:
        if self.model.sizes[layer]["h"] != self.model.sizes[layer]["c"]:
            c2h_wo_proj = self.model.lstm.weight_P[layer]

            self.decompositions[layer]["h"][:, :, i] = (
                self.decompositions[layer]["h_wo_proj"][:, :, i] @ c2h_wo_proj
            )

        else:
            h_wo_proj = self.decompositions[layer]["h_wo_proj"][:, :, i]
            self.decompositions[layer]["h"][:, :, i] = h_wo_proj

    def _add_interactions(
        self, layer: int, i: int, gate: Tensor, source: Tensor, cell_type: str = "c"
    ) -> None:
        """ Allows for interactions to be grouped differently than as specified in the paper. """
        if gate.shape == source.shape:
            source_bias = source[:, -1, :]
            source_signal = source[:, :-1, :]
        else:
            source_bias = None
            source_signal = source

        for p_idx in range(self.num_partitions):
            gated_source = gate * source_signal[:, p_idx]
            self.decompositions[layer][cell_type][:, p_idx, i] += gated_source.sum(
                dim=1
            )

        if source_bias is not None:
            init_partition = self.idx2partition_idx[0]
            gated_bias = gate * source_bias
            self.decompositions[layer][cell_type][
                :, init_partition, i
            ] += gated_bias.sum(dim=1)

    def _reset_decompositions(self, batch_size: int, slen: int) -> None:
        sizes = self.model.sizes

        # The h_wo_proj decompositions are used for lstms with differing hidden and cell sizes.
        # The h decomposition is then created by the projection of `h_wo_proj` in `_project_hidden`.
        self.decompositions = {
            layer: {
                "c": torch.zeros(
                    (batch_size, self.num_partitions, slen + 1, sizes[layer]["c"]),
                    dtype=config.DTYPE,
                ),
                "h": torch.zeros(
                    (batch_size, self.num_partitions, slen + 1, sizes[layer]["c"]),
                    dtype=config.DTYPE,
                ),
                "h_wo_proj": torch.zeros(
                    (batch_size, self.num_partitions, slen + 1, sizes[layer]["c"]),
                    dtype=config.DTYPE,
                ),
            }
            for layer in range(self.model.num_layers)
        }

        init_partition = self.idx2partition_idx[0]
        for layer in range(self.model.num_layers):
            self.decompositions[layer]["c"][
                :, init_partition, 0
            ] = self.activation_dict[layer, "icx"]
            self.decompositions[layer]["h"][
                :, init_partition, 0
            ] = self.activation_dict[layer, "ihx"]

    def _set_input_partitions(
        self, input_partitions: Optional[List[InputPartition]], slen: int
    ) -> None:
        """ Creates partition set for which relevance is calculated. """
        if input_partitions is None:
            # 0 index indicates initial states, first input should be addressed with 1
            self.num_partitions = slen + 1
            self.idx2partition_idx = {idx: idx for idx in range(slen + 1)}
        else:
            self.num_partitions = len(input_partitions) + 1
            for idx in range(slen + 1):
                for p_idx, partition in enumerate(input_partitions, start=1):
                    if (isinstance(partition, tuple) and idx in range(*partition)) or (
                        isinstance(partition, list) and idx in partition
                    ):
                        assert (
                            idx not in self.idx2partition_idx
                        ), f"Input partitions are overlapping for item index {idx}!"
                        self.idx2partition_idx[idx] = p_idx

                        # If 0 (i.e., init states + biases) is provided explicitly we don't need to
                        # add a special partition for these interactions.
                        if idx == 0:
                            self.num_partitions -= 1

                # If an index is not part of any partition we add it to the "drain" partition that
                # contains all other indices, indexed by 0.
                if idx not in self.idx2partition_idx:
                    if idx == 0:
                        self.idx2partition_idx[0] = 0
                    else:
                        self.idx2partition_idx[idx] = self.idx2partition_idx[0]

    def _split_model_bias(self, batch_size: int) -> None:
        for layer in range(self.model.num_layers):
            bias = self.model.bias[layer]

            self.bias.update(
                {
                    (layer, name): tensor.repeat((batch_size, 1, 1))
                    for name, tensor in zip(
                        self.model.split_order, torch.chunk(bias, 4, dim=0)
                    )
                }
            )

            self.bias[layer, "f"] += self.model.forget_offset

    def _validate_decomposition(self) -> None:
        """ Sum of decomposed state should equal original hidden state. """
        true_hidden = self.activation_dict[self.toplayer, "hx"]
        dec_hidden = self.decompositions[self.toplayer]["h"][:, :, 1:].sum(1)

        avg_difference = torch.mean(true_hidden - dec_hidden)
        max_difference = torch.max(torch.abs(true_hidden - dec_hidden))

        assert torch.sum(torch.isnan(avg_difference)) == 0, "State contains nan values"

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

    def _calc_scores(self) -> Tensor:
        decomposition = self.decompositions[self.toplayer]["h"]

        # TODO: check if still needed
        # if self.extra_classes is not None:
        #     for i, j in enumerate(self.extra_classes, start=1):
        #         rel_dec[:, -i] = rel_dec[:, j]
        #         irrel_dec[:, -i] = irrel_dec[:, j]

        # If no classes have been provided the decomposed states themselves are returned
        if self.decoder_b.size(1) == 0:
            return decomposition

        num_classes = self.decoder_w.size(2)
        scores = torch.zeros(
            (self.batch_size, self.num_partitions, self.slen + 1, num_classes),
            dtype=config.DTYPE,
        )
        for p_idx in range(self.num_partitions):
            scores[:, p_idx] = torch.bmm(decomposition[:, p_idx], self.decoder_w)

        return scores
