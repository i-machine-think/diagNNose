"""
Module to define a mechanism superclass to provide "interventions", or more specifically an adjustment of activations
in an RNNs during processing time based on some sort of intervention signal.
"""

from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable, Dict, Tuple

from torch import Tensor

from diagnnose.model_wrappers.intervention_lstm import InterventionLSTM
from diagnnose.typedefs.activations import ActivationTensors


class InterventionMechanism(ABC):
    """
    A callable class that is being used as an decorator function to
    provide an intervention functionality when
    wrapped around model's forward() function.

    Example usage:
    >> mechanism = InterventionMechanism(model, ...)
    >> model = mechanism.apply()

    Parameters
    ----------
    model: InterventionLSTM
        Model to which the mechanism is being applied to.
    trigger_func: Callable
        Function that triggers an intervention, either by returning a mask or a step size.

    """

    def __init__(self, model: InterventionLSTM, trigger_func: Callable):
        self.model = model
        self.trigger_func = trigger_func

    def __call__(self, forward_func: Callable) -> Callable:
        """Wrap the intervention function about the models forward
        function and return the decorated function.

        Parameters
        ----------
        forward_func: Callable
            Forward function of the model the mechanism is applied to.

        Returns
        -------
        wrapped: Callable:
            Decorated forward function.
        """

        @wraps(forward_func)
        def wrapped(
            inp: str, prev_activations: ActivationTensors, **additional: Dict
        ) -> Tuple[Tensor, ActivationTensors]:

            out, activations = forward_func(inp, prev_activations)

            return self.intervention_func(
                inp, prev_activations, out, activations, **additional
            )

        return wrapped

    def apply(self) -> InterventionLSTM:
        """ Return an instance of the model where the intervention
        function decorates the model's forward function.

        Returns
        -------
        model : InterventionLSTM
            Model with intervention mechanism applied to it.
        """
        self.model.forward = self(self.model.forward)  # Decorate forward function
        return self.model

    @abstractmethod
    def intervention_func(
        self,
        inp: str,
        prev_activations: ActivationTensors,
        out: Tensor,
        activations: ActivationTensors,
        **additional: Dict
    ) -> Tuple[Tensor, ActivationTensors]:
        """
        Define the intervention logic here.

        Parameters
        ----------
        inp: str
            Current input token.
        prev_activations: FullActivationDict
            Activations of the previous time step.
        out: Tensor
            Output Tensor of current time step.
        activations: FullActivationDict
            Activations of current time step,
        additional: dict
            Dictionary of additional information delivered via keyword arguments.
        """
        # Use self.trigger_func on input arguments here to determine when to trigger an intervention
        # Afterwards match the return signature of the original model forward() function
        ...
