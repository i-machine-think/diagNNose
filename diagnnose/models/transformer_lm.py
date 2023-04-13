from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import log_softmax
from torchtext.data import Batch

from diagnnose.attribute import ShapleyTensor
from diagnnose.models import LanguageModel
from diagnnose.typedefs.activations import (
    ActivationDict,
    ActivationName,
    ActivationNames,
    SelectionFunc,
)


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
        ``"(distil)-(Ro)BERT(a)"``, ``"(distil)-gpt2"``, ``"XLM"``
    cache_dir: str, optional
        Path towards the cache directory where the HF model weights will
        be stored.
    compute_pseudo_ll: bool, optional
        Toggle to compute the Pseudo Log-Likelihood that was introduced
        by Salazar et al. (2020). This can be used to compute the
        sentence probabilies of bi-directional masked LMs, masking out
        one token at the time.
    device : str, optional
        Torch device on which forward passes will be run.
        Defaults to cpu.
    """

    def __init__(
        self,
        embeddings_attr: Optional[str] = None,
        compute_pseudo_ll: bool = False,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(device)

        self.pretrained_model = self.load_model(**kwargs)

        self.embeddings_attr = embeddings_attr
        self.compute_pseudo_ll = compute_pseudo_ll

    def load_model(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def embeddings(self) -> Callable[[Tensor], Tensor]:
        raise NotImplementedError

    @property
    def decoder(self) -> nn.Module:
        raise NotImplementedError

    def base_model(self, compute_out: bool):
        return self.pretrained_model

    def forward(
        self,
        input_ids: Optional[Union[Tensor, List[int]]] = None,
        inputs_embeds: Optional[Union[Tensor, ShapleyTensor]] = None,
        input_lengths: Optional[List[int]] = None,
        attention_mask: Optional[Union[Tensor, List[int]]] = None,
        compute_out: bool = True,
        calc_causal_lm_probs: bool = False,
        only_return_top_embs: bool = True,
        mask_idx: Optional[int] = None,
        selection_func: Optional[SelectionFunc] = None,
        batch: Optional[Batch] = None,
    ) -> Union[ActivationDict, Tensor]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        if inputs_embeds is None and input_ids is None:
            raise ValueError("inputs_embeds or input_ids must be provided")
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(self.device)
            if len(inputs_embeds.shape) == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)  # Add batch dimension
        if input_lengths is None:
            if input_ids is not None:
                batch_size, max_sen_len = input_ids.shape
            else:
                batch_size, max_sen_len = inputs_embeds.shape[:2]
            input_lengths = torch.tensor(batch_size * [max_sen_len], device=self.device)
        if isinstance(attention_mask, list):
            attention_mask = torch.tensor(attention_mask, device=self.device)
        if attention_mask is None:
            attention_mask = self.create_attention_mask(input_lengths)

        model = self.base_model(compute_out)
        attention_mask = attention_mask.to(self.device)

        activation_name = (-1, "out") if compute_out else (-1, "hx")

        if self.compute_pseudo_ll:
            assert isinstance(mask_idx, int), "mask_idx must be provided for Pseudo LL"
            if inputs_embeds is None:
                inputs_embeds = self.create_inputs_embeds(input_ids)

            logits = self._forward_pseudo_ll(
                model,
                inputs_embeds,
                attention_mask,
                mask_idx,
                activation_name,
                compute_out,
                selection_func=selection_func,
                batch=batch,
            )
        else:
            logits = self._forward(
                model,
                compute_out,
                attention_mask,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
            )

        if calc_causal_lm_probs:
            output_ids = input_ids[:, 1:].unsqueeze(-1)
            probs = log_softmax(logits[:, :-1], dim=-1)
            logits = torch.gather(probs, -1, output_ids)

        if only_return_top_embs:
            return logits

        return {activation_name: logits}

    @staticmethod
    def _forward(
        model,
        compute_out: bool,
        attention_mask: Tensor,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        output = model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        if hasattr(output, "logits"):
            logits: Tensor = output.logits
        elif hasattr(output, "last_hidden_state"):
            logits = output.last_hidden_state
        elif isinstance(output, tuple):
            # Fairseq output logic
            if compute_out:
                logits = output[0]
            else:
                logits = output[1]["inner_states"][-1].transpose(0, 1)
                logits = model.decoder.layer_norm(logits)
        else:
            raise AttributeError

        return logits

    def _forward_pseudo_ll(
        self,
        model,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        mask_idx: int,
        activation_name: ActivationName,
        compute_out: bool,
        selection_func: Optional[SelectionFunc] = None,
        batch: Optional[Batch] = None,
    ) -> Tensor:
        mask_embedding = self.embeddings(torch.tensor(mask_idx, device=self.device))

        max_sen_len = inputs_embeds.shape[1]

        pseudo_ll_logits = torch.zeros(
            *inputs_embeds.shape[:2], self.nhid(activation_name), device=self.device
        )

        sen_column = batch.dataset.sen_column
        sen_lens = getattr(batch, sen_column)[1]
        for w_idx in range(max_sen_len):
            if selection_func is not None:
                sen_ids = []
                for batch_idx, (sen_idx, sen_len) in enumerate(
                    zip(batch.sen_idx, sen_lens)
                ):
                    if (w_idx < sen_len) and selection_func(
                        w_idx, batch.dataset.examples[sen_idx]
                    ):
                        sen_ids.append(batch_idx)
                if len(sen_ids) == 0:
                    continue
            else:
                sen_ids = [
                    batch_idx
                    for batch_idx, sen_len in enumerate(sen_lens)
                    if w_idx < sen_len
                ]

            masked_inputs_embeds = inputs_embeds[sen_ids].clone()
            masked_inputs_embeds[:, w_idx] = mask_embedding
            masked_attention_mask = attention_mask[sen_ids].clone()

            logits = self._forward(
                model,
                compute_out,
                masked_attention_mask,
                inputs_embeds=masked_inputs_embeds,
            )
            pseudo_ll_logits[sen_ids, w_idx] = logits[:, w_idx]

        return pseudo_ll_logits

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

    def create_inputs_embeds(self, input_ids: Union[Tensor, List[int]]) -> Tensor:
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, device=self.device)

        inputs_embeds = self.embeddings(input_ids)

        return inputs_embeds

    @property
    def num_layers(self) -> int:
        return self.pretrained_model.config.n_layer

    @property
    def top_layer(self) -> int:
        return -1

    def nhid(self, activation_name: ActivationName) -> int:
        model_config = self.pretrained_model.config

        if activation_name[1] == "out":
            return model_config.vocab_size
        elif hasattr(model_config, "word_embed_proj_dim"):
            return model_config.word_embed_proj_dim
        else:
            return self.pretrained_model.config.hidden_size

    @staticmethod
    def activation_names() -> ActivationNames:
        return [(-1, "out")]
