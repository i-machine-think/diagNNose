"""
Module to define a mechanism superclass to provide "interventions", or more specifically an adjustment of activations
in an RNNs during processing time based on some sort of intervention signal.
"""

import abc
from functools import wraps
from typing import Callable, Tuple

from torch import Tensor

from models.forward_lstm import ForwardLSTM
from typedefs.models import FullActivationDict


class InterventionMechanism:
    """
    A callable class that is being used as an decorator function to provide an intervention functionality when
    wrapped around model's forward() function.

    Example usage:
    >> mechanism = InterventionMechanism(model, ...)
    >> model = mechanism.apply()
    """
    def __init__(self,
                 model: ForwardLSTM,
                 trigger_func: Callable):
        self.model = model
        self.trigger_func = trigger_func

    def __call__(self,
                 forward_func: Callable) -> Callable:
        """
        Wrap the intervention function about the models forward function and return the decorated function.
        """
        @wraps(forward_func)
        def wrapped(inp: str,
                    prev_activations: FullActivationDict) -> Tuple[Tensor, FullActivationDict]:
            return self.intervention_func(inp, prev_activations)

        return wrapped

    def apply(self) -> ForwardLSTM:
        """
        Return an instance of the model where the intervention function decorates the model's forward function.
        """
        self.model.forward = self(self.model.forward)  # Decorate forward function
        return self.model

    @abc.abstractmethod
    def intervention_func(self,
                          inp: str,
                          prev_activations: FullActivationDict) -> Tuple[Tensor, FullActivationDict]:
        """
        Define the intervention logic here.
        """
        # Use self.trigger func on input arguments here to determine when to trigger an intervention
        # Afterwards match the return signature of the original model forward() function
        ...



