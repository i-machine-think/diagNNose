from abc import ABC, abstractmethod
from typing import Optional, Union

import torch.nn as nn
from torch import Tensor

from diagnnose.attribute import ShapleyTensor
from diagnnose.typedefs.activations import (
    ActivationDict,
    ActivationName,
    ActivationNames,
    SizeDict,
)


class LanguageModel(ABC, nn.Module):
    sizes: SizeDict = {}
    is_causal: bool = False

    def __init__(self, device: str = "cpu"):
        super().__init__()

        self.device = device

    @abstractmethod
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Union[Tensor, ShapleyTensor]] = None,
        input_lengths: Optional[Tensor] = None,
        compute_out: bool = False,
        calc_causal_lm_probs: bool = False,
        only_return_top_embs: bool = False,
    ) -> Union[ActivationDict, Tensor]:
        """Performs a single forward pass across all LM layers.

        Parameters
        ----------
        input_ids : Tensor, optional
            Indices of input sequence tokens in the vocabulary.
            Size: batch_size x max_sen_len
        inputs_embeds : Tensor | ShapleyTensor, optional
            This is useful if you want more control over how to convert
            `input_ids` indices into associated vectors than the model's
            internal embedding lookup matrix. Also allows a
            ShapleyTensor to be provided, allowing feature contributions
            to be track during a forward pass.
            Size: batch_size x max_sen_len x nhid
        input_lengths : Tensor, optional
            Tensor of the sentence lengths of each batch item.
            If not provided, all items are assumed the same length.
            Size: batch_size,
        compute_out : bool, optional
            Toggles the computation of the final decoder projection.
            If set to False this projection is not calculated.
            Defaults to False.
        calc_causal_lm_probs : bool, optional
            Toggle to directly compute the output probabilities of the
            next token within the forward pass. Next token is inferred
            based on the shifted input tokens. Should only be toggled
            for Causal (a.k.a. auto-regressive) LMs. Defaults to False.
        only_return_top_embs : bool, optional
            Toggle to only return the tensor of the hidden state of the
            top layer of the network, instead of the full activation
            dictionary. Defaults to False.

        Returns
        -------
        activations : ActivationDict
            Dictionary mapping an activation name to a (padded) tensor.
            Size: a_name -> batch_size x max_sen_len x nhid
        """
        raise NotImplementedError

    @abstractmethod
    def create_inputs_embeds(self, input_ids: Tensor) -> Tensor:
        """Transforms a sequence of input tokens to their embedding.

        Parameters
        ----------
        input_ids : Tensor
            Tensor of shape batch_size x max_sen_len.

        Returns
        -------
        inputs_embeds : Tensor
            Embedded tokens of shape batch_size x max_sen_len x nhid.
        """
        raise NotImplementedError

    @abstractmethod
    def activation_names(self) -> ActivationNames:
        """Returns a list of all the model's activation names.

        Returns
        -------
        activation_names : ActivationNames
            List of (layer, name) tuples.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """ Returns the number of layers in the LM. """
        raise NotImplementedError

    @property
    @abstractmethod
    def top_layer(self) -> int:
        """ Returns the index of the LM's top layer. """
        raise NotImplementedError

    @abstractmethod
    def nhid(self, activation_name: ActivationName) -> int:
        """ Returns number of hidden units for a (layer, name) tuple. """
        raise NotImplementedError
