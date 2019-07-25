from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from overrides import overrides
from torch import Tensor, nn

from diagnnose.typedefs.activations import TensorDict


SizeDict = Dict[int, Dict[str, int]]


class LanguageModel(ABC, nn.Module):
    device: str = "cpu"
    forget_offset: int = 0
    ih_concat_order: List[str] = ["h", "i"]
    sizes: SizeDict = {}
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
        input_ : Tensor
            Input batch, presumed to be mapped already to token id's
            by `torchtext`.
        prev_activations : TensorDict
            Dict mapping each (layer, 'hx'/'cx') tuple to a tensor:
            {layer => {'hx'|'cx' => torch.Tensor}}
        compute_out : bool, optional
            Allows to skip softmax calculation, defaults to True.

        Returns
        -------
        out : torch.Tensor
            Torch Tensor of output distribution of vocabulary
        activations : TensorDict
            Dictionary mapping each layer to each activation name to a tensor
        """

    @abstractmethod
    def init_hidden(self, bsz: int) -> TensorDict:
        """"""
