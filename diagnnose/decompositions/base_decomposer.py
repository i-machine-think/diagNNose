from typing import Any, List, Optional, Union

import torch
from torch import Tensor

from diagnnose.typedefs.activations import ActivationDict, ActivationName, NamedTensors
from diagnnose.typedefs.classifiers import LinearDecoder
from diagnnose.typedefs.models import LanguageModel


class BaseDecomposer:
    """ Base decomposer which decomposition classes should inherit

    Parameters
    ----------
    model : LanguageModel
        LanguageModel for which decomposition will be performed
    activation_dict : PartialArrayDict
        Dictionary containing the necessary activations for decomposition
    decoder : (Tensor, Tensor)
        (Coefficients, bias) tuple of the (linear) decoding layer.
        Expected dimensions: ((num_classes, hidden_dim), (hidden_dim,)).
    final_index : Tensor
        1-d Tensor with index of final element of a batch element.
        Due to masking for sentences of uneven length the final index
        can differ between batch elements.
    """

    def __init__(
        self,
        model: LanguageModel,
        activation_dict: ActivationDict,
        decoder: LinearDecoder,
        final_index: Tensor,
        extra_classes: List[int],
    ) -> None:
        self.model = model
        self.weights: NamedTensors = {}
        self.bias: ActivationDict = {}
        self.decoder_w, self.decoder_b = decoder

        self.activation_dict = activation_dict

        self.final_index = final_index
        self.batch_size = final_index.size(0)
        self.slen = max(final_index).item() + 1
        self.extra_classes = extra_classes

        self._validate_activation_shapes()

    def decompose(self, *arg: Any, **kwargs: Any) -> Union[NamedTensors, Tensor]:
        raise NotImplementedError

    def decompose_with_bias(
        self, *arg: Any, **kwargs: Any
    ) -> Union[Tensor, NamedTensors]:
        decomposition = self.decompose(*arg, **kwargs)

        bias = self.decompose_bias()
        bias = torch.repeat_interleave(bias.unsqueeze(0), 2, dim=0).unsqueeze(1)
        for key, arr in decomposition.items():
            decomposition[key] = torch.cat((arr, bias), dim=1)

        return decomposition

    def decompose_bias(self) -> Tensor:
        return torch.exp(self.decoder_b)

    def calc_original_logits(self, normalize: bool = False) -> Tensor:

        assert (
            self.model.top_layer,
            "hx",
        ) in self.activation_dict, (
            "'hx' should be provided to calculate the original logit"
        )
        final_hidden_state = self.get_final_activations((self.model.top_layer, "hx"))

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

    def _split_model_bias(self, batch_size: Optional[int] = None) -> None:
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

            self.bias[layer, "f"] += self.model.forget_offset

        if batch_size is not None:
            for a_name, tensor in self.bias.items():
                self.bias[a_name] = tensor.repeat((batch_size, 1))

    def _validate_activation_shapes(self) -> None:
        pass
