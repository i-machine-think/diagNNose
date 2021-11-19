import abc
from typing import TYPE_CHECKING, Dict, List, Optional, Type

import torch
from torch import Tensor
from transformers import BatchEncoding

from .gcd_tensor import GCDTensor
from .shapley_tensor import ShapleyTensor

if TYPE_CHECKING:
    from diagnnose.models import LanguageModel


tensor_types: Dict[str, Type[ShapleyTensor]] = {
    "ShapleyTensor": ShapleyTensor,
    "GCDTensor": GCDTensor,
}


class Decomposer(abc.ABC):
    """Abstract base class for Decomposer classes.

    A Decomposer takes care of dividing the input features into the
    desired partition of contributions.
    """

    def __init__(
        self,
        model: "LanguageModel",
        num_samples: Optional[int] = None,
        tensor_type: str = "ShapleyTensor",
    ):
        self.model = model
        self.num_samples = num_samples
        self.tensor_type = tensor_types[tensor_type]

    @abc.abstractmethod
    def decompose(self, batch_encoding: BatchEncoding) -> ShapleyTensor:
        raise NotImplementedError

    @abc.abstractmethod
    def wrap_inputs_embeds(self, input_ids: Tensor) -> ShapleyTensor:
        raise NotImplementedError


class ShapleyDecomposer(Decomposer):
    """A ShapleyDecomposer propagates all input feature contributions
    simultaneously.

    That is, an input sequence of :math:`n` features will be transformed
    into a ShapleyTensor containing :math:`n` feature contributions.

    Concretely: if we have an input tensor :math:`X` of shape:
    ``(num_features, input_dim)`` we can express this as a sum of
    features:
    :math:`X = \\sum_i^n \\phi^i`, where :math:`\\phi^i` is also of
    shape ``(num_features, input_dim)``, with
    :math:`\\phi^i_j =
    \\begin{cases}X_j&i=j\\\\0&\\textit{otherwise}\\end{cases}`

    Without approximations this way of partitioning scales
    exponentially in the number of input features, quickly becoming
    infeasible when :math:`n > 10`.
    """

    def decompose(self, batch_encoding: BatchEncoding) -> ShapleyTensor:
        input_ids = torch.tensor(batch_encoding["input_ids"])
        inputs_embeds = self.wrap_inputs_embeds(input_ids)

        with torch.no_grad():
            shapley_out = self.model(
                inputs_embeds=inputs_embeds,
                input_lengths=batch_encoding.data.get("length", None),
                compute_out=True,
                only_return_top_embs=True,
            )

        return shapley_out

    def wrap_inputs_embeds(self, input_ids: Tensor) -> ShapleyTensor:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)

        # Shape: batch_size x max_sen_len x nhid
        inputs_embeds = self.model.create_inputs_embeds(input_ids)

        # First contribution corresponds to contributions stemming from bias terms within the
        # model itself.
        contributions = [torch.zeros_like(inputs_embeds)]

        # Each individual contribution is set to its corresponding input feature, and set to
        # zero on all other positions.
        for w_idx in range(inputs_embeds.shape[1]):
            contribution = torch.zeros_like(inputs_embeds)
            contribution[:, w_idx] = inputs_embeds[:, w_idx]
            contributions.append(contribution)

        shapley_in = self.tensor_type(
            inputs_embeds,
            contributions=contributions,
            validate=True,
            num_samples=self.num_samples,
        )

        return shapley_in


class ContextualDecomposer(Decomposer):
    """A ContextualDecomposer propagates each input feature
    contribution individually, set out against the contributions of all
    other features combined.

    This idea has been proposed in Murdocht et al., (2018):
    https://arxiv.org/abs/1801.05453

    An input sequence of :math:`n` features will be transformed
    into a ShapleyTensor containing :math:`2` feature contributions:
    one containing the contributions of the feature of interest
    (:math:`\\beta`), and one containing the contributions of all
    other features (:math:`\\gamma`).

    Concretely: if we have an input tensor :math:`X` of shape:
    ``(num_features, input_dim)`` we can express this as a sum of
    features:
    :math:`X = \\beta^i + \\gamma^i`, where both :math:`\\beta` and
    :math:`\\gamma` are also of shape ``(num_features, input_dim)``,
    with :math:`\\beta^i_j =
    \\begin{cases}X_j&i=j\\\\0&\\textit{otherwise}\\end{cases}` and
    :math:`\\gamma^i_j =
    \\begin{cases}X_j&i\\neq j\\\\0&\\textit{otherwise}\\end{cases}`

    This way of partitioning scales polynomially in the number of input
    features, but requires a separate forward pass for each individual
    feature contribution :math:`\\beta^i`.
    """

    def decompose(self, batch_encoding: BatchEncoding) -> ShapleyTensor:
        input_ids = torch.tensor(batch_encoding["input_ids"])
        shapley_tensors = self.wrap_inputs_embeds(input_ids)

        all_contributions = []

        for w_idx, inputs_embeds in enumerate(shapley_tensors):
            with torch.no_grad():
                out, (beta, _gamma) = self.model(
                    inputs_embeds=inputs_embeds,
                    input_lengths=batch_encoding.data.get("length", None),
                    compute_out=True,
                    only_return_top_embs=True,
                )
            all_contributions.append(beta)

        return GCDTensor(out, all_contributions)

    def wrap_inputs_embeds(self, input_ids: Tensor) -> List[ShapleyTensor]:
        assert (
            input_ids.ndim == 2
        ), "Input ids must contain both batch and sentence dimension"

        inputs_embeds = self.model.create_inputs_embeds(input_ids)

        all_shapley_in = [
            GCDTensor(
                inputs_embeds,
                contributions=[torch.zeros_like(inputs_embeds), inputs_embeds],
                validate=True,
                num_samples=self.num_samples,
            )
        ]

        for w_idx in range(inputs_embeds.shape[1]):
            beta = torch.zeros_like(inputs_embeds)
            gamma = inputs_embeds.clone()

            beta[:, w_idx] = gamma[:, w_idx]
            gamma[:, w_idx] = 0.0

            contributions = [torch.zeros_like(beta), beta, gamma]

            shapley_in = GCDTensor(
                inputs_embeds,
                contributions=contributions,
                validate=True,
                num_samples=self.num_samples,
                baseline_partition=0,
            )

            all_shapley_in.append(shapley_in)

        return all_shapley_in
