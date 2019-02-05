from typing import Tuple

from overrides import overrides
from torch import Tensor, nn

from ..typedefs.models import FullActivationDict


class LanguageModel(nn.Module):
    """ Abstract class for LM with intermediate activations """
    def __init__(self) -> None:
        super(LanguageModel, self).__init__()

    @overrides
    def forward(self,
                inp: str,
                prev_activations: FullActivationDict) -> Tuple[Tensor, FullActivationDict]:
        """

        Parameters
        ----------
        inp : str
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
        raise NotImplementedError
