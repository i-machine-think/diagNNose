from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from overrides import overrides
from torch import Tensor, nn

from diagnnose.activations.init_states import InitStates
from diagnnose.typedefs.activations import TensorDict


SizeDict = Dict[int, Dict[str, int]]


class LanguageModel(ABC, nn.Module):
    """ Abstract class for LM with intermediate activations """

    device: str = "cpu"
    forget_offset: int = 0
    ih_concat_order: List[str] = ["h", "i"]
    init_states: Optional[InitStates] = None
    sizes: SizeDict = {}
    split_order: List[str]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    @property
    def num_layers(self) -> int:
        return len(self.sizes)

    @overrides
    @abstractmethod
    def forward(
        self, input_: Tensor, prev_activations: TensorDict, compute_out: bool = True
    ) -> Tuple[Optional[Tensor], TensorDict]:
        """

        Parameters
        ----------
        input_ : Tensor
            Tensor containing a batch of token id's at the current
            sentence position.
        prev_activations : TensorDict, optional
            Dict mapping the activation names of the previous hidden
            and cell states to their corresponding Tensors. Defaults to
            None, indicating the initial states will be used.
        compute_out : bool, optional
            Toggles the computation of the final decoder projection.
            If set to False this projection is not calculated.
            Defaults to True.

        Returns
        -------
        out : torch.Tensor, optional
            Torch Tensor of output distribution of vocabulary. If
            `compute_out` is set to True, `out` returns None.
        activations : TensorDict
            Dictionary mapping each layer to each activation name to a
            tensor.
        """

    def init_hidden(self, bsz: int) -> TensorDict:
        """"""
        assert self.init_states is not None, "Initial states must be provided"

        return self.init_states.create(bsz)
