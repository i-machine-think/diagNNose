from typing import Dict

from overrides import overrides

from diagnnose.model_wrappers.forward_lstm import ForwardLSTM
from diagnnose.typedefs.activations import ActivationTensors


class InterventionLSTM(ForwardLSTM):
    """
    Variant of the ForwardLSTM that allows to pass more keyword arguments during the forward pass, e.g. labels
    or other information that can be used during an intervention. See interventions.mechanism and
    interventions.weakly_supervised for more details.
    """

    @overrides
    def forward(
        self, inp: str, prev_activations: ActivationTensors, **additional: Dict
    ):
        """
        Parameters
        ----------
        inp : str
            input token that is mapped to id
        prev_activations : FullActivationDict
            Dictionary mapping each layer to 'hx' and 'cx' to a tensor:
            {layer => {'hx'|'cx' => torch.Tensor}}
        additional: dict
            Dictionary of additional information delivered via keyword arguments.

        Returns
        -------
        out : torch.Tensor
            Torch Tensor of output distribution of vocabulary
        activations : FullActivationDict
            Dictionary mapping each layer to each activation name to a tensor
        """
        # **additional is not used here but can be accessed by InterventionMechanism through a function decorator.
        # See interventions.mechanism.InterventionMechanism.__call__
        return super().forward(inp, prev_activations)
