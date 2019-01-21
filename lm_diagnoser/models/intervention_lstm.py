from typing import Dict

from overrides import overrides

from models import ForwardLSTM
from typedefs import FullActivationDict


class InterventionLSTM(ForwardLSTM):
    """
    Variant of the ForwardLSTM that allows to pass more keyword arguments during the forward pass, e.g. labels
    or other information that can be used during an intervention. See interventions.mechanism and
    interventions.weakly_supervised for more details.
    """
    @overrides
    def forward(self,
                inp: str,
                prev_activations: FullActivationDict,
                **additional: Dict):

        # **additional is not used here but can be accessed by InterventionMechanism through a function decorator.
        # See interventions.mechanism.InterventionMechanism.__call__
        return super().forward(inp, prev_activations)
