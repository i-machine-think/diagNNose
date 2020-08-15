from typing import List, Optional

import torch
from torch import Tensor

from diagnnose.models import LanguageModel

from .shapley_tensor import ShapleyTensor


class Decomposer:
    def __init__(self, model: LanguageModel):
        self.model = model

    def decompose(
        self, input_ids: Tensor, attention_mask: Optional[Tensor]
    ) -> ShapleyTensor:
        shapley_in = self.wrap_inputs_embeds(input_ids)

        with torch.no_grad():
            shapley_out = self.model(
                inputs_embeds=shapley_in, attention_mask=attention_mask
            )

        return shapley_out

    def wrap_inputs_embeds(self, input_ids: Tensor) -> ShapleyTensor:
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


class ContextualDecomposer(Decomposer):
    def decompose(
        self, input_ids: Tensor, attention_mask: Optional[Tensor]
    ) -> ShapleyTensor:
        shapley_tensors = self.wrap_inputs_embeds(input_ids)

        contributions = []

        for w_idx, inputs_embeds in enumerate(shapley_tensors):
            with torch.no_grad():
                out, c = self.model(
                    inputs_embeds=inputs_embeds, attention_mask=attention_mask
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
