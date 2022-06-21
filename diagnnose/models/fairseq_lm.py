from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from fairseq.models import FairseqLanguageModel
from fairseq.models.transformer import Embedding, TransformerDecoder
from fairseq.tasks.language_modeling import LanguageModelingTask
from torch import Tensor

from .transformer_lm import TransformerLM


class FairseqPretrainedModel(FairseqLanguageModel):
    def __init__(self, decoder):
        super().__init__(decoder)

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        embed_tokens = cls.build_embedding(
            task.source_dictionary, args.decoder_input_dim
        )

        decoder = TransformerDecoder(
            args, task.target_dictionary, embed_tokens, no_encoder_attn=True
        )
        return cls(decoder)

    @classmethod
    def build_embedding(cls, dictionary, embed_dim):
        embed_tokens = Embedding(len(dictionary), embed_dim, dictionary.pad())
        return embed_tokens

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        assert input_ids is not None

        return self.decoder(input_ids)


@dataclass
class TaskArgs:
    data: str
    output_dictionary_size: int


@dataclass
class FairseqConfig:
    vocab_size: int
    hidden_size: int
    n_layer: int


class FairseqLM(TransformerLM):
    is_causal = True

    def load_model(self, checkpoint: str, vocab_dir: str, **kwargs):
        checkpoint = torch.load(checkpoint)
        checkpoint_args = checkpoint["cfg"]["model"]

        task_args = TaskArgs(
            vocab_dir,
            checkpoint["cfg"]["model"].output_dictionary_size,
        )
        task = LanguageModelingTask.setup_task(task_args)
        model = FairseqPretrainedModel.build_model(checkpoint_args, task)
        model.load_state_dict(checkpoint["model"])

        config = FairseqConfig(
            model.decoder.embed_tokens.num_embeddings,
            model.decoder.output_embed_dim,
            model.decoder.num_layers,
        )
        model.config = config

        return model

    @property
    def embeddings(self) -> Callable[[Tensor], Tensor]:
        return self.pretrained_model.decoder.embed_tokens

    @property
    def decoder(self) -> nn.Module:
        return self.pretrained_model.decoder.output_projection

    @property
    def decoder_w(self) -> nn.Module:
        return self.decoder.weight.data
