from functools import reduce
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch import Tensor
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    DistilBertForMaskedLM,
    XLMWithLMHeadModel,
)

from .transformer_lm import TransformerLM

mode_to_auto_model = {
    "causal_lm": AutoModelForCausalLM,
    "masked_lm": AutoModelForMaskedLM,
    "question_answering": AutoModelForQuestionAnswering,
    "sequence_classification": AutoModelForSequenceClassification,
    "token_classification": AutoModelForTokenClassification,
}


class HuggingfaceLM(TransformerLM):
    def load_model(
        self,
        transformer_type: str,
        mode: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        auto_model = mode_to_auto_model.get(mode, AutoModel)

        self.is_causal = mode == "causal_lm"

        return auto_model.from_pretrained(transformer_type, cache_dir=cache_dir)

    def base_model(self, compute_out: bool):
        if compute_out:
            return self.pretrained_model

        return self.pretrained_model.base_model

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
            elif hasattr(base_model, "word_emb"):
                # Transformer-XL
                return base_model.word_emb
            elif hasattr(base_model, "decoder"):
                # OPT
                return base_model.decoder.embed_tokens
            else:
                raise AttributeError("word embedding attribute not found")

    @property
    def decoder(self) -> nn.Module:
        # RoBERTa / BERT / XLNet / ALBERT
        for attr in ["lm_head", "cls", "lm_loss", "predictions"]:
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
