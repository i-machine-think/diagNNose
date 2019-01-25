from typing import Tuple
from ..typedefs.models import FullActivationDict
from overrides import overrides

from torch import nn, Tensor


class LanguageModel(nn.Module):
    """ Abstract class for LM with intermediate activations """
    def __init__(self) -> None:
        super(LanguageModel, self).__init__()

    @overrides
    def forward(self,
                inp: str,
                prev_activations: FullActivationDict) -> Tuple[Tensor, FullActivationDict]:
        """

        Args:
            inp: input token that is mapped to id
            prev_activations: {layer => {'hx'|'cx' => torch.Tensor}}

        Returns:
            out: Torch Tensor of output distribution of vocabulary
            activations: Dict of all intermediate activations
        """
        raise NotImplementedError
