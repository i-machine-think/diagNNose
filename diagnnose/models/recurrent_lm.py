from typing import List

from torch import Tensor

from diagnnose.models import LanguageModel
from diagnnose.typedefs.activations import ActivationDict, ActivationName


class RecurrentLM(LanguageModel):
    """ Abstract class for RNN LM with intermediate activations """

    forget_offset: int = 0
    ih_concat_order: List[str] = ["h", "i"]
    split_order: List[str]
    use_char_embs: bool = False
    init_states: ActivationDict = {}

    @property
    def num_layers(self) -> int:
        return max(layer for layer, _name in self.sizes) + 1

    @property
    def top_layer(self) -> int:
        return self.num_layers - 1

    @property
    def output_size(self) -> int:
        return self.sizes[self.top_layer, "hx"]

    def init_hidden(self, batch_size: int) -> ActivationDict:
        """Creates a batch of initial states.

        Parameters
        ----------
        batch_size : int
            Size of batch for which states are created.

        Returns
        -------
        init_states : ActivationTensors
            Dictionary mapping hidden and cell state to init tensors.
        """
        batch_init_states: ActivationDict = {}

        for layer in range(self.num_layers):
            for hc in ["hx", "cx"]:
                # Shape: (batch_size, nhid)
                batched_state = self.init_states[layer, hc].repeat(batch_size, 1)
                batch_init_states[layer, hc] = batched_state

        return batch_init_states

    def final_hidden(self, hidden: ActivationDict) -> Tensor:
        """Returns the final hidden state.

        Parameters
        ----------
        hidden : ActivationTensors
            Dictionary of extracted activations.

        Returns
        -------
        final_hidden : Tensor
            Tensor of the final hidden state.
        """
        return hidden[self.top_layer, "hx"].squeeze()

    def nhid(self, activation_name: ActivationName) -> int:
        """Returns number of hidden units for a (layer, name) tuple.

        If `name` != emb/hx/cx returns the size of (layer, `cx`).
        """
        layer, name = activation_name

        return self.sizes.get((layer, name), self.sizes[layer, "cx"])
