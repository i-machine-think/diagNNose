from typing import List

import torch
from torch import Tensor
from transformers import BatchEncoding

from diagnnose.attribute.shapley_tensor import ShapleyTensor
from diagnnose.models import LanguageModel


class ShapleyDecomposer:
    def __init__(self, model: LanguageModel):
        self.model = model

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

        shapley_in = ShapleyTensor(
            inputs_embeds, contributions=contributions, validate=True
        )

        return shapley_in


class ContextualDecomposer(ShapleyDecomposer):
    def decompose(self, batch_encoding: BatchEncoding) -> ShapleyTensor:
        input_ids = torch.tensor(batch_encoding["input_ids"])
        shapley_tensors = self.wrap_inputs_embeds(input_ids)

        contributions = []

        for w_idx, inputs_embeds in enumerate(shapley_tensors):
            with torch.no_grad():
                out, c = self.model(
                    inputs_embeds=inputs_embeds,
                    input_lengths=batch_encoding.data.get("length", None),
                    compute_out=True,
                    only_return_top_embs=True,
                )
            beta = c[0] if w_idx == 0 else c[1]
            contributions.append(beta)

        return ShapleyTensor(out, contributions)

    def wrap_inputs_embeds(self, input_ids: Tensor) -> List[ShapleyTensor]:
        inputs_embeds = self.model.create_inputs_embeds(input_ids)

        all_shapley_in = [
            ShapleyTensor(
                inputs_embeds,
                contributions=[torch.zeros_like(inputs_embeds), inputs_embeds],
                validate=True,
            )
        ]

        for w_idx in range(inputs_embeds.shape[1]):
            beta = torch.zeros_like(inputs_embeds)
            gamma = inputs_embeds.clone()

            beta[:, w_idx] = gamma[:, w_idx]
            gamma[:, w_idx] = 0.0

            contributions = [torch.zeros_like(inputs_embeds), beta, gamma]

            shapley_in = ShapleyTensor(
                inputs_embeds, contributions=contributions, validate=True
            )

            all_shapley_in.append(shapley_in)

        return all_shapley_in
