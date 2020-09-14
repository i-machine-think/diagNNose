from functools import reduce
from typing import List, Optional, Union

import torch
from torch import Tensor
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    PreTrainedModel,
)

from diagnnose.attribute.shapley_tensor import ShapleyTensor
from diagnnose.models import LanguageModel
from diagnnose.typedefs.activations import (
    ActivationDict,
    ActivationName,
    ActivationNames,
)

mode_to_auto_model = {
    None: AutoModel,
    "language_modeling": AutoModelForMaskedLM,
    "question_answering": AutoModelForQuestionAnswering,
    "sequence_classification": AutoModelForSequenceClassification,
    "token_classification": AutoModelForTokenClassification,
}


class TransformerLM(LanguageModel):
    """ Huggingface LM wrapper. """

    def __init__(
        self,
        model_name: str,
        mode: Optional[str] = None,
        embeddings_attr: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()

        auto_model = mode_to_auto_model[mode]

        self.model: PreTrainedModel = auto_model.from_pretrained(
            model_name, cache_dir=cache_dir
        )

        self.embeddings_attr = embeddings_attr

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Union[Tensor, ShapleyTensor]] = None,
        input_lengths: Optional[List[int]] = None,
        attention_mask: Optional[Tensor] = None,
        compute_out: bool = True,
        only_return_top_embs: bool = True,
    ) -> Union[ActivationDict, Tensor]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        if inputs_embeds is None and input_ids is None:
            raise ValueError("inputs_embeds or input_ids must be provided")
        if inputs_embeds is None:
            inputs_embeds = self.create_inputs_embeds(input_ids)
        if len(inputs_embeds.shape) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)  # Add batch dimension
        if input_lengths is None:
            batch_size, max_sen_len = inputs_embeds.shape[:2]
            input_lengths = torch.tensor(batch_size * [max_sen_len])
        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_lengths)

        model = self.model if compute_out else self.model.base_model
        output = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        if isinstance(output, tuple) and only_return_top_embs:
            output = output[0]

        return output

    @staticmethod
    def create_attention_mask(input_lengths: List[int]) -> Tensor:
        """Creates an attention mask as described in:
        https://huggingface.co/transformers/glossary.html#attention-mask

        Parameters
        ----------
        input_lengths : List[int]
            List containing sentence lengths of each batch item.

        Returns
        -------
        attention_mask : Tensor
            Attention mask prescribing which items may be taken into
            account by the attention mechanism.
            Size: batch_size x max_sen_length
        """
        max_sen_len = max(input_lengths)

        attention_mask = torch.zeros(len(input_lengths), max_sen_len)

        for idx, length in enumerate(input_lengths):
            attention_mask[idx, :length] = 1.0

        return attention_mask

    def create_inputs_embeds(self, input_ids: Tensor) -> Tensor:
        if self.embeddings_attr is not None:
            attrs = self.embeddings_attr.split(".")
            embeddings = reduce(getattr, attrs, self.model)
            inputs_embeds: Tensor = embeddings(input_ids)
        else:
            base_model = self.model.base_model
            if hasattr(base_model, "wte"):
                # GPT-2
                inputs_embeds: Tensor = base_model.wte(input_ids)
            elif hasattr(base_model, "embeddings"):
                if hasattr(base_model.embeddings, "word_embeddings"):
                    # BERT-based models, Electra, Longformer, Reformer
                    inputs_embeds = base_model.embeddings.word_embeddings(input_ids)
                else:
                    # XLM
                    inputs_embeds = base_model.embeddings(input_ids)
            elif hasattr(base_model, "word_embedding"):
                # XLNet
                inputs_embeds = base_model.word_embedding(input_ids)
            elif hasattr(base_model, "w"):
                # CTRL
                inputs_embeds = base_model.w(input_ids)
            elif hasattr(base_model, "encoder"):
                # T5
                inputs_embeds = base_model.encoder.embed_tokens(input_ids)
            else:
                raise AttributeError("word embedding attribute not found")

        return inputs_embeds

    def num_layers(self) -> int:
        return self.model.config.n_layer

    def top_layer(self) -> int:
        return self.model.config.n_layer - 1

    def nhid(self, activation_name: ActivationName) -> int:
        return self.model.config.hidden_size

    def activation_names(self) -> ActivationNames:
        return [(-1, "hx")]
