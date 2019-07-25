from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from overrides import overrides
from torch import Tensor, nn

from diagnnose.typedefs.activations import TensorDict


class LanguageModel(ABC, nn.Module):
    array_type: str
    device: str = "cpu"
    forget_offset: int = 0
    ih_concat_order: List[str] = ["h", "i"]
    sizes: Dict[int, Dict[str, int]] = {}
    split_order: List[str]

    """ Abstract class for LM with intermediate activations """

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    @property
    def num_layers(self) -> int:
        return len(self.sizes)

    @overrides
    @abstractmethod
    def forward(
        self, input_: Tensor, prev_activations: TensorDict, compute_out: bool = True
    ) -> Tuple[Tensor, TensorDict]:
        """

        Parameters
        ----------
        token : str
            input token that is mapped to id
        prev_activations : FullActivationDict
            Dictionary mapping each layer to 'hx' and 'cx' to a tensor:
            {layer => {'hx'|'cx' => torch.Tensor}}
        compute_out : bool, optional
            Allows to skip softmax calculation, defaults to True.

        Returns
        -------
        out : torch.Tensor
            Torch Tensor of output distribution of vocabulary
        activations : FullActivationDict
            Dictionary mapping each layer to each activation name to a tensor
        """
