from functools import reduce
from typing import Callable, List, Optional, Union

import torch
from torch import Tensor
from torch.nn.functional import log_softmax
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    DistilBertForMaskedLM,
    PreTrainedModel,
    XLMWithLMHeadModel,
)

from diagnnose.attribute import ShapleyTensor
from diagnnose.models import LanguageModel
from diagnnose.typedefs.activations import (
    ActivationDict,
    ActivationName,
    ActivationNames,
)

mode_to_auto_model = {
    "causal_lm": AutoModelForCausalLM,
    "masked_lm": AutoModelForMaskedLM,
    "question_answering": AutoModelForQuestionAnswering,
    "sequence_classification": AutoModelForSequenceClassification,
    "token_classification": AutoModelForTokenClassification,
}


class TransformerLM(LanguageModel):
    """Huggingface LM wrapper.

    Parameters
    ----------
    transformer_type: str
        Transformer type that can be passed to
        ``auto_model.from_pretrained`` as one of the valid models made
        available by Huggingface, e.g. ``"roberta-base"``.
    mode : str, optional
        Language model mode, one of ``"causal_lm"``, ``"masked_lm"``,
        ``"question_answering"``, ``"sequence_classification"``, or
        ``"token_classification"``. If not provided the model will be
        imported using ``AutoModel``, which often yields an LM with no
        task-specific head on top.
    embeddings_attr : str, optional
        Attribute name of the word embeddings of the model. Can be
        nested. For example, if the word embeddings are stored as an
        ``"wte"`` attribute that is part of the ``"encoder"`` attribute
        of the full model, you would pass ``"encoder.wte"``. For the
        following models this parameter does not need to be passed:
        ``"(distil)-(Ro)BERT(a)"``, ``"(distil)-gpt2"``, ``"XLM"``, ``""``, ``""``,
    device : str, optional
        Torch device on which forward passes will be run.
        Defaults to cpu.
    """

    def __init__(
        self,
        transformer_type: str,
        mode: Optional[str] = None,
        embeddings_attr: Optional[str] = None,
        cache_dir: Optional[str] = None,
        device: str = "cpu",
    ):
        super().__init__()

        auto_model = mode_to_auto_model.get(mode, AutoModel)

        self.pretrained_model: PreTrainedModel = auto_model.from_pretrained(
            transformer_type, cache_dir=cache_dir
        )

        self.embeddings_attr = embeddings_attr
        self.device = device
        self.is_causal = mode == "causal_lm"

    def forward(
        self,
        input_ids: Optional[Union[Tensor, List[int]]] = None,
        inputs_embeds: Optional[Union[Tensor, ShapleyTensor]] = None,
        input_lengths: Optional[List[int]] = None,
        attention_mask: Optional[Union[Tensor, List[int]]] = None,
        compute_out: bool = True,
        calc_causal_lm_probs: bool = False,
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
            input_lengths = torch.tensor(batch_size * [max_sen_len], device=self.device)
        if isinstance(attention_mask, list):
            attention_mask = torch.tensor(attention_mask, device=self.device)
        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_lengths)

        model = (
            self.pretrained_model if compute_out else self.pretrained_model.base_model
        )
        inputs_embeds = inputs_embeds.to(self.device)
        attention_mask = attention_mask.to(self.device)

        output = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        if hasattr(output, "logits"):
            logits: Tensor = output.logits
            activation_name = (-1, "out")
        elif hasattr(output, "last_hidden_state"):
            logits = output.last_hidden_state
            activation_name = (-1, "hx")
        else:
            raise AttributeError

        if calc_causal_lm_probs:
            output_ids = input_ids[:, 1:].unsqueeze(-1)
            probs = log_softmax(logits[:, :-1], dim=-1)
            logits = torch.gather(probs, -1, output_ids)

        if only_return_top_embs:
            return logits

        return {activation_name: logits}

    def create_attention_mask(self, input_lengths: List[int]) -> Tensor:
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

        attention_mask = torch.zeros(
            len(input_lengths), max_sen_len, device=self.device
        )

        for idx, length in enumerate(input_lengths):
            attention_mask[idx, :length] = 1.0

        return attention_mask

    @property
    def embeddings(self) -> Callable[[Tensor], Tensor]:
        if self.embeddings_attr is not None:
            attrs = self.embeddings_attr.split(".")
            return reduce(getattr, attrs, self.pretrained_model)
        else:
            base_model = self.pretrained_model.base_model
            if hasattr(base_model, "wte"):
                # GPT-2
                return base_model.wte
            elif hasattr(base_model, "embeddings"):
                if hasattr(base_model.embeddings, "word_embeddings"):
                    # BERT-based models, Electra, Longformer, Reformer
                    return base_model.embeddings.word_embeddings
                else:
                    # XLM
                    return base_model.embeddings
            elif hasattr(base_model, "word_embedding"):
                # XLNet
                return base_model.word_embedding
            elif hasattr(base_model, "w"):
                # CTRL
                return base_model.w
            elif hasattr(base_model, "encoder"):
                # T5
                return base_model.encoder.embed_tokens
            else:
                raise AttributeError("word embedding attribute not found")

    def create_inputs_embeds(self, input_ids: Union[Tensor, List[int]]) -> Tensor:
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, device=self.device)

        inputs_embeds = self.embeddings(input_ids)

        return inputs_embeds

    @property
    def decoder(self) -> torch.nn.Module:
        # RoBERTa / BERT
        for attr in ["lm_head", "cls"]:
            if hasattr(self.pretrained_model, attr):
                return getattr(self.pretrained_model, attr)

        if isinstance(self.pretrained_model, DistilBertForMaskedLM):
            return torch.nn.Sequential(
                self.pretrained_model.vocab_transform,
                torch.nn.GELU(),
                self.pretrained_model.vocab_layer_norm,
                self.pretrained_model.vocab_projector,
            )

        if isinstance(self.pretrained_model, XLMWithLMHeadModel):
            return self.pretrained_model.pred_layer.proj

        raise AttributeError("Model decoder not found")

    @property
    def num_layers(self) -> int:
        return self.pretrained_model.config.n_layer

    @property
    def top_layer(self) -> int:
        return -1

    def nhid(self, activation_name: ActivationName) -> int:
        if activation_name[1] == "out":
            return self.pretrained_model.config.vocab_size

        return self.pretrained_model.config.hidden_size

    def activation_names(self) -> ActivationNames:
        return [(-1, "out")]
