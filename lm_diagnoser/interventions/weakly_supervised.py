"""
Implementing a version of the intervention mechanism which relies on weak supervision, i.e. additional labels which
provide task-relevant information.
"""

from abc import abstractmethod, ABC
from typing import Dict, Tuple
import re

import numpy as np
from sklearn.linear_model import LogisticRegressionCV as LogReg
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
from torch.nn.functional import sigmoid
from torch.optim import SGD
from overrides import overrides

from ..interventions.mechanism import InterventionMechanism
from ..models.forward_lstm import ForwardLSTM
from ..typedefs.interventions import DiagnosticClassifierDict
from ..typedefs.models import FullActivationDict


class WeaklySupervisedInterventionMechanism(InterventionMechanism, ABC):
    """
    Mechanism that triggers an intervention based on additional labels and Diagnostic classifiers [1][2].

    [1] https://www.jair.org/index.php/jair/article/view/11196/26408
    [2] https://arxiv.org/abs/1808.08079

    Parameters
    ----------
    model: InterventionLSTM
        Model to which the mechanism is being applied to.

    Attributes
    ----------
    step_size: float
        Step size of the adjustment of the activations.
    diagnostic_classifiers: dict
        Dictionary of path to diagnostic classifiers to their respective diagnostic classifiers objects.
    topmost_layer: str
        Name of the topmost RNN layer.
    num_topmost_layer: int
        Number of the topmost RNN layer.
    """
    def __init__(self,
                 model: ForwardLSTM,
                 diagnostic_classifiers: DiagnosticClassifierDict,
                 step_size: float):

        super().__init__(model, trigger_func=self.dc_trigger_func)
        self.step_size = step_size

        # Link diagnostic classifiers to layer they correspond to
        self.diagnostic_classifiers = {
            re.search('(l\d+)', path).group(0): dc for path, dc in diagnostic_classifiers.items()
        }
        # Apply interventions only to topmost layer
        self.topmost_layer = sorted(self.diagnostic_classifiers.keys())[-1]
        self.num_topmost_layer = len(self.diagnostic_classifiers) - 1

    @abstractmethod
    def select_diagnostic_classifier(self,
                                     inp: str,
                                     prev_activations: FullActivationDict,
                                     **additional: dict):
        """
        Select the appropriate Diagnostic Classifier based on data used in current forward pass.

        Parameters
        ----------
        inp: str
            Current input token.
        prev_activations: FullActivationDict
            Activations of the previous time step.
        additional: dict
            Dictionary of additional information delivered via keyword arguments.
        """
        ...

    @abstractmethod
    def dc_trigger_func(self,
                        prev_activations: FullActivationDict,
                        activations: FullActivationDict,
                        out: Tensor,
                        prediction: Tensor,
                        **additional: dict) -> Tensor:
        """
        Use a Diagnostic Classifier to determine for which batch instances to trigger an intervention.
        Returns a binary mask with 1 corresponding to an impending intervention.

        Parameters
        ----------
        prev_activations: FullActivationDict
            Activations of the previous time step.
        out: Tensor
            Output Tensor of current time step.
        prediction: Tensor
            Prediction of the diagnostic classifier based on the current time step's activations.
        activations: FullActivationDict
            Activations of current time step,
        additional: dict
            Dictionary of additional information delivered via keyword arguments.
        """
        ...

    def diagnostic_classifier_to_vars(self,
                                      diagnostic_classifier: LogReg) -> Tuple[Tensor, Tensor]:
        """
        Convert the weights and bias of a Diagnostic Classifier (trained with scikit-learn) into PyTorch Variables.

        Parameters
        ----------
        diagnostic_classifier: LogisticRegressionCV
            Diagnostic classifier trained with scikit-learn on RNN activations.

        Returns
        -------
        weights: Variable
            Diagnostic classifier weights as PyTorch variable.
        bias: Variable
            Diagnostic classifier bias. as PyTorch variable.
        """
        weights = self._wrap_in_var(diagnostic_classifier.coef_, requires_grad=False)
        bias = self._wrap_in_var(diagnostic_classifier.intercept_, requires_grad=False)

        return weights, bias

    @staticmethod
    def _wrap_in_var(array: np.array,
                     requires_grad: bool) -> Variable:
        """
        Wrap a numpy array into a PyTorch Variable.

        Parameters
        ----------
        array: np.array
            Numpy array to be converted to a PyTorch Variable.
        requires_grad: bool
            Whether the variable requires the calculation of its gradients.
        """
        return Variable(torch.tensor(array, dtype=torch.float).squeeze(0), requires_grad=requires_grad)

    @abstractmethod
    def diagnostic_classifier_loss(self,
                                   prediction: Tensor,
                                   label: Tensor) -> _Loss:
        """
        Define in this function how the loss of the Diagnostic Classifier's prediction w.r.t to the loss is calculated.
        Should return a subclass of PyTorch's _Loss object like NLLLoss or CrossEntropyLoss.

        Parameters
        ----------
        prediction: Tensor
            Prediction of the diagnostic classifier based on the current time step's activations.
        label: Tensor
            Actual label to compare the prediction to.
        """
        ...

    @overrides
    def intervention_func(self,
                          inp: str,
                          prev_activations: FullActivationDict,
                          out: Tensor,
                          activations: FullActivationDict,
                          **additional: Dict) -> Tuple[Tensor, FullActivationDict]:
        """
        Conduct an intervention based on weak supervision signal.

        Parameters
        ----------
        inp: str
            Current input token.
        prev_activations: FullActivationDict
            Activations of the previous time step.
        out: Tensor
            Output Tensor of current time step.
        activations: FullActivationDict
            Activations of current time step.
        additional: dict
            Dictionary of additional information delivered via keyword arguments.

        Returns
        -------
        out: Tensor
            Re-decoded output Tensor of current time step.
        activations: FullActivationDict
            Activations of current time step after interventions.
        """

        dc = self.select_diagnostic_classifier(inp, prev_activations, **additional)
        weights, bias = self.diagnostic_classifier_to_vars(dc)
        label = torch.tensor([additional["label"]], dtype=torch.float)

        # Calculate gradient of the diagnostic classifier's prediction w.r.t. the current activations
        current_activations = activations[self.num_topmost_layer]["hx"]
        current_activations = self._wrap_in_var(current_activations, requires_grad=True)
        params = [current_activations]
        optimizer = SGD(params, lr=self.step_size)
        optimizer.zero_grad()

        prediction = sigmoid(weights @ current_activations + bias).unsqueeze(0)
        mask = self.dc_trigger_func(prev_activations, activations, out, prediction, **additional)
        loss = self.diagnostic_classifier_loss(prediction, label)
        loss.backward()
        gradient = current_activations.grad

        # Manual (masked) update step
        new_activations = current_activations + self.step_size * gradient * mask

        # Repeat decoding step with adjusted activations
        out: Tensor = self.model.w_decoder @ new_activations + self.model.b_decoder

        return out, activations
