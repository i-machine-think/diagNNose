from typing import Any, List, Optional

import torch
from torch import Tensor

from diagnnose.models.lm import LanguageModel
from diagnnose.typedefs.activations import (
    ActivationName,
    ActivationTensors,
    NamedTensors,
)
from diagnnose.typedefs.classifiers import LinearDecoder


class BaseDecomposer:
    """ Base decomposer which decomposition classes should inherit

    Parameters
    ----------
    model : LanguageModel
        LanguageModel for which decomposition will be performed
    activation_dict : PartialArrayDict
        Dictionary containing the necessary activations for decomposition
    decoder : (Tensor, Tensor) ((num_classes, hidden_dim), (hidden_dim,))
        (Coefficients, bias) tuple of the (linear) decoding layer
    final_index : Tensor
        1-d Tensor with index of final element of a batch element.
        Due to masking for sentences of uneven length the final index
        can differ between batch elements.
    """

    def __init__(
        self,
        model: LanguageModel,
        activation_dict: ActivationTensors,
        decoder: LinearDecoder,
        final_index: Tensor,
        extra_classes: List[int],
    ) -> None:
        self.model = model
        self.decoder_w, self.decoder_b = decoder
        self.activation_dict = activation_dict

        self.final_index = final_index
        self.batch_size = final_index.size(0)
        self.toplayer = model.num_layers - 1
        self.extra_classes = extra_classes

        self._validate_activation_shapes()
        self._append_init_states()

    def _decompose(self, *arg: Any, **kwargs: Any) -> NamedTensors:
        raise NotImplementedError

    def decompose(
        self, *arg: Any, append_bias: bool = False, **kwargs: Any
    ) -> NamedTensors:
        decomposition = self._decompose(*arg, **kwargs)

        if append_bias:
            bias = self.decompose_bias()
            bias = torch.repeat_interleave(bias.unsqueeze(0), 2, dim=0).unsqueeze(1)
            for key, arr in decomposition.items():
                decomposition[key] = torch.cat((arr, bias), dim=1)

        return decomposition

    def decompose_bias(self) -> Tensor:
        return torch.exp(self.decoder_b)

    def calc_original_logits(self, normalize: bool = False) -> Tensor:
        assert (
            self.toplayer,
            "hx",
        ) in self.activation_dict, (
            "'hx' should be provided to calculate the original logit"
        )
        final_hidden_state = self.get_final_activations((self.toplayer, "hx"))

        original_logit = torch.exp(
            (final_hidden_state @ self.decoder_w.t()) + self.decoder_b
        )

        if normalize:
            original_logit = (
                torch.exp(original_logit).t()
                / torch.sum(torch.exp(original_logit), dim=1)
            ).t()

        return original_logit

    def get_final_activations(self, a_name: ActivationName, offset: int = 0) -> Tensor:
        return self.activation_dict[a_name][
            range(self.batch_size), self.final_index + offset
        ]

    def _validate_activation_shapes(self) -> None:
        pass

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

                    if cell_type == "hx" and layer == self.toplayer:
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

                        if cell_type == "hx" and layer == self.toplayer:
                            self.final_index += 1
