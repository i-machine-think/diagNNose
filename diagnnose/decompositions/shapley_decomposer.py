import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from overrides import overrides
from torch import Tensor
from scipy.special import expit as sigmoid

import diagnnose.typedefs.config as config
from diagnnose.typedefs.activations import Decompositions, NamedTensors

from .contextual_decomposer import BaseDecomposer
from .shapley_full import calc_full_shapley_values


# Tuple: range(start, end) or list of indices
# First token of sentence is index 1, init states are index 0
InputPartition = Union[Tuple[int, int], List[int]]


class ShapleyDecomposer(BaseDecomposer):
    """ Shapley Decomposer, uses the mechanism of CD but considers all
    input features simultaneously.

    Inherits and uses functions from BaseDecomposer.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.activations: NamedTensors = {}
        self.decompositions: Decompositions = {}

        self.idx2partition_idx: Dict[int, int] = {}
        self.num_partitions: int = 0

        self.gate_bias_rel = True

    @overrides
    def decompose(
        self,
        input_partitions: Optional[List[InputPartition]] = None,
        gate_bias_rel: bool = True,
    ) -> Tensor:
        """ Main loop for the contextual decomposition.

        Parameters
        ----------
        input_partitions : List[InputPartition], optional
            Optional list of input partitions: represented as either
            a tuple of (start, stop) spans, or lists of indices that
            should be grouped together. If not provided each individual
            input will stand on its own.
        gate_bias_rel : bool, optional
            When using the relevance partition of Arras et al. (see
            thesis for details) the gates don't need to be decomposed.
            If we want separate splits for different gate interactions
            this toggle can be set to False. Defaults to True.

        Returns
        -------
        scores : NamedArrayDict
            Dictionary with keys `relevant` and `irrelevant`, containing
            the decomposed scores for the earlier provided decoder.
        """
        self._split_model_bias(self.batch_size)
        self._set_input_partitions(input_partitions, self.slen)
        self._reset_decompositions(self.batch_size, self.slen)

        self.gate_bias_rel = gate_bias_rel

        for layer in range(self.model.num_layers):
            for i in range(1, self.slen + 1):
                self._calc_activations(layer, i)

                self._add_forget_decomposition(layer, i)

                self._add_input_decomposition(layer, i)

                self._add_output_decomposition(layer, i)

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
            # Shape: (batch_size, num_partitions)
            embs = self.activation_dict[0, "emb"][:, i - 1]
            nhid = embs.size(1)
            input_ = torch.zeros(
                (self.batch_size, self.num_partitions, nhid), dtype=config.DTYPE
            )
            p_idx = self.idx2partition_idx[i]
            input_[:, p_idx] = embs
        else:
            input_ = self.decompositions[layer - 1]["h"][:, :, i]

        # TODO: future partitions need not to be taken into account here yet, saves computation
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
        if self.gate_bias_rel:
            # Shape: (batch_size, nhid)
            shapley_f = self.activation_dict[layer, "f_g"][:, i - 1]
        else:
            # Shape: (batch_size, num_partitions+1, nhid)
            shapley_f = self._calc_shapley_values(layer, i, "f")

        # Shape: (batch_size, num_partitions, nhid)
        prev_c = self.decompositions[layer]["c"][:, :, i - 1]

        self._add_interactions(layer, i, shapley_f, prev_c)

    def _add_input_decomposition(self, layer: int, i: int) -> None:
        """ Calculates the input gate decomposition, Equation (17) of the paper. """
        if self.gate_bias_rel:
            # Shape: (batch_size, nhid)
            shapley_i = self.activation_dict[layer, "i_g"][:, i - 1]
        else:
            # Shape: (batch_size, num_partitions+1, nhid)
            shapley_i = self._calc_shapley_values(layer, i, "i")

        # Shape: (batch_size, num_partitions + 1, nhid)
        shapley_g = self._calc_shapley_values(layer, i, "g")

        self._add_interactions(layer, i, shapley_i, shapley_g)

    def _add_output_decomposition(self, layer: int, i: int) -> None:
        """ Calculates the output gate decomposition, Equation (23). """
        if self.gate_bias_rel:
            # Shape: (batch_size, nhid)
            shapley_o = self.activation_dict[layer, "o_g"][:, i - 1]
        else:
            # Shape: (batch_size, num_partitions, nhid)
            c = self.decompositions[layer]["c"][:, :, i]
            if hasattr(self.model, "peepholes"):
                self.activations["o"] += c * self.model.peepholes[layer, "o"]

            # Shape: (batch_size, num_partitions + 1, nhid)
            shapley_o = self._calc_shapley_values(layer, i, "o")

        # Shape: (batch_size, num_partitions, nhid)
        shapley_c = self._calc_shapley_values(layer, i, "c")

        self._add_interactions(layer, i, shapley_o, shapley_c, cell_type="h_wo_proj")

    def _calc_shapley_values(self, layer: int, i: int, activation_type: str) -> Tensor:
        """ Calculate the Shapley values at a specific instance.

        We only need to take into account the activations of the
        partitions that have been observed up to step i, which saves a
        lot of computation.
        """
        partitions_seen = self._partitions_seen(i)

        contains_bias = activation_type != "c"

        if contains_bias:
            # Shape: (batch_size, num_partitions + 1, nhid)
            activations = self.activations[activation_type]
            bias = self.bias[layer, activation_type].unsqueeze(1)
            activations = torch.cat((activations, bias), dim=1)
            partitions_seen.append(-1)
        else:
            # Shape: (batch_size, num_partitions, nhid)
            activations = self.decompositions[layer]["c"][:, :, i]

        shapley_values = torch.zeros(activations.shape, dtype=config.DTYPE)
        activations = activations[:, partitions_seen]
        func = np.tanh if activation_type in ["c", "g"] else sigmoid

        # Shape: (batch_size, num_partitions (+ 1), nhid)
        shapley_values[:, partitions_seen] = calc_full_shapley_values(
            activations, func, bias_as_baseline=contains_bias
        )

        return shapley_values

    def _partitions_seen(self, i: int) -> List[int]:
        """ Return the set of partitions that have been seen at step i.
        Future partitions to not need to be taken into account yet.
        """
        first_i_partitions = list(self.idx2partition_idx.values())[: i + 1]

        return list(set(first_i_partitions))

    def _add_interactions(
        self,
        layer: int,
        i: int,
        gate: Tensor,
        source: Tensor,
        cell_type: str = "c",
    ) -> None:
        """ Allows for interactions to be grouped differently than as specified in the paper. """

        init_partition = self.idx2partition_idx[0]

        # Add bias to init/bias partition
        if source.size(1) != self.num_partitions:
            source[:, init_partition] += source[:, -1]
            source = source[:, :-1]

        if not self.gate_bias_rel:
            gate_bias = gate[:, -1]
            gate = gate[:, :-1].sum(1)

            self.decompositions[layer][cell_type][:, init_partition, i] += gate_bias * source.sum(1)

        for p_idx in range(self.num_partitions):
            gated_source = gate * source[:, p_idx]

            self.decompositions[layer][cell_type][:, p_idx, i] += gated_source

    def _project_hidden(self, layer: int, i: int) -> None:
        if self.model.sizes[layer]["h"] != self.model.sizes[layer]["c"]:
            c2h_wo_proj = self.model.lstm.weight_P[layer]

            self.decompositions[layer]["h"][:, :, i] = (
                self.decompositions[layer]["h_wo_proj"][:, :, i] @ c2h_wo_proj
            )

        else:
            h_wo_proj = self.decompositions[layer]["h_wo_proj"][:, :, i]
            self.decompositions[layer]["h"][:, :, i] = h_wo_proj

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

        # Set initial states at first position of init states partition
        init_p_idx = self.idx2partition_idx[0]
        for layer in range(self.model.num_layers):
            self.decompositions[layer]["c"][:, init_p_idx, 0] = self.activation_dict[
                layer, "icx"
            ]
            self.decompositions[layer]["h"][:, init_p_idx, 0] = self.activation_dict[
                layer, "ihx"
            ]

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
                # TODO: if init p_idx is provided, start=1 causes partition 0 to be unused
                for p_idx, partition in enumerate(input_partitions, start=1):
                    if (isinstance(partition, tuple) and idx in range(*partition)) or (
                        isinstance(partition, list) and idx in partition
                    ):
                        assert (
                            idx not in self.idx2partition_idx
                        ), f"Input partitions are overlapping for item index {idx}!"
                        self.idx2partition_idx[idx] = p_idx

                # If an index is not part of any partition we add it to the "drain" partition that
                # contains all other indices, indexed by 0.
                if idx not in self.idx2partition_idx:
                    if idx == 0:
                        self.idx2partition_idx[0] = 0
                    else:
                        self.idx2partition_idx[idx] = self.idx2partition_idx[0]

    def _validate_decomposition(self) -> None:
        """ Sum of decomposed state should equal original hidden state. """
        true_hidden = self.activation_dict[self.model.top_layer, "hx"]
        dec_hidden = self.decompositions[self.model.top_layer]["h"][:, :, 1:].sum(1)

        avg_difference = torch.mean(true_hidden - dec_hidden)
        max_difference = torch.max(torch.abs(true_hidden - dec_hidden))

        if torch.sum(torch.isnan(avg_difference)) != 0:
            warnings.warn("Decomposed state contains NaN values.")

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
        decomposition = self.decompositions[self.model.top_layer]["h"]

        # If no classes have been provided the decomposed states themselves are returned
        if self.decoder_b.size(1) == 0:
            # Shape: (bach_size, num_partitions, slen + 1, nhid)
            return decomposition

        num_classes = self.decoder_w.size(2)
        scores = torch.zeros(
            (self.batch_size, self.num_partitions, self.slen + 1, num_classes),
            dtype=config.DTYPE,
        )
        for p_idx in range(self.num_partitions):
            scores[:, p_idx] = torch.bmm(decomposition[:, p_idx], self.decoder_w)

        # Shape: (batch_size, num_partitions, slen + 1, num_classes)
        return scores
