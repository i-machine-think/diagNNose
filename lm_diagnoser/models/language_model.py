from typing import Dict, Tuple
from overrides import overrides

from torch import nn, Tensor


ActivationLayer = Dict[str, Tensor]
ActivationDict = Dict[int, ActivationLayer]
ParameterDict = ActivationDict


class LanguageModel(nn.Module):
    """ Abstract class for LM with intermediate activations """
    def __init__(self) -> None:
        super(LanguageModel, self).__init__()

    @overrides
    def forward(self,
                inp: str,
                prev_activations: ActivationDict) -> Tuple[Tensor, ActivationDict]:
        """

        Args:
            inp: input token that is mapped to id
            prev_activations: {layer => {'hx'|'cx' => torch.Tensor}}

        Returns:
            out: Torch Tensor of output distribution of vocabulary
            activations: Dict of all intermediate activations
        """
        raise NotImplementedError
