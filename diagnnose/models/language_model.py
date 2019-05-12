from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from overrides import overrides
from torch import Tensor, nn

from diagnnose.typedefs.activations import FullActivationDict


class LanguageModel(ABC, nn.Module):
    num_layers: int
    hidden_size_c: int
    hidden_size_h: int
    split_order: List[str]
    array_type: str

    """ Abstract class for LM with intermediate activations """
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

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
