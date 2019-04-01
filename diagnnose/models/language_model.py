from abc import ABC, abstractmethod
from typing import Tuple

from overrides import overrides
from torch import Tensor, nn

from ..typedefs.activations import FullActivationDict


class LanguageModel(ABC, nn.Module):
    """ Abstract class for LM with intermediate activations """
    @abstractmethod
    def __init__(self,
                 model: str,
                 vocab: str,
                 lm_module: str,
                 device: str = 'cpu') -> None:
        super().__init__()

        self.num_layers: int = 0
        self.hidden_size: int = 0

    @overrides
    @abstractmethod
    def forward(self,
                token: str,
                prev_activations: FullActivationDict) -> Tuple[Tensor, FullActivationDict]:
        """

        Parameters
        ----------
        token : str
            input token that is mapped to id
        prev_activations : FullActivationDict
            Dictionary mapping each layer to 'hx' and 'cx' to a tensor:
            {layer => {'hx'|'cx' => torch.Tensor}}

        Returns
        -------
        out : torch.Tensor
            Torch Tensor of output distribution of vocabulary
        activations : FullActivationDict
            Dictionary mapping each layer to each activation name to a tensor
        """
