from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor

from diagnnose.typedefs.activations import ActivationDict, ActivationNames


class LanguageModel(ABC, nn.Module):
    @abstractmethod
    def forward(
        self, batch: Tensor, batch_lengths: Tensor, compute_out: bool = True
    ) -> ActivationDict:
        """ Performs a single forward pass across all LM layers.

        Parameters
        ----------
        batch : Tensor
            Tensor of a batch of (padded) sentences.
            Size: batch_size x max_sen_len
        batch_lengths : Tensor
            Tensor of the sentence lengths of each batch item.
            Size: batch_size
        compute_out : bool, optional
            Toggles the computation of the final decoder projection.
            If set to False this projection is not calculated.
            Defaults to True.

        Returns
        -------
        activations : ActivationDict
            Dictionary mapping an activation name to a (padded) tensor.
            Size: a_name -> batch_size x max_sen_len x nhid
        """
        raise NotImplementedError

    @abstractmethod
    def activation_names(self) -> ActivationNames:
        """ Returns a list of all the model's activation names.

        Parameters
        ----------

        Returns
        -------
        activation_names : ActivationNames
            List of (layer, name) tuples.
        """
        raise NotImplementedError
