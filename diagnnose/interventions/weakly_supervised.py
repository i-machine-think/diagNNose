"""
Implementing a version of the intervention mechanism which relies on weak supervision, i.e. additional labels which
provide information that is helpful to solve the task or enrich the model in any other way.

NOTE: THIS MODULE HAS NOT BEEN MAINTAINED
"""

import re
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
from overrides import overrides
from sklearn.linear_model import LogisticRegressionCV as LogReg
from torch import Tensor
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
from torch.optim import SGD

from diagnnose.interventions.mechanism import InterventionMechanism
from diagnnose.model_wrappers.forward_lstm import ForwardLSTM
from diagnnose.typedefs.activations import ActivationTensors
from diagnnose.typedefs.interventions import DiagnosticClassifierDict


class WeaklySupervisedMechanism(InterventionMechanism, ABC):
    """
    Mechanism that triggers an intervention based on additional labels
    and Diagnostic classifiers [1][2].

    [1] https://www.jair.org/index.php/jair/article/view/11196/26408
    [2] http://aclweb.org/anthology/W18-5426

    Parameters
    ----------
    model: InterventionLSTM
        Model to which the mechanism is being applied to.

    Attributes
    ----------
    step_size: float
        Step size of the adjustment of the activations.
    masking: bool
        Flag to indicate whether interventions should only be conducted
        when the prediction of the Diagnostic Classifier
        is wrong (masking = True) or even when it is right (masking =
        False; in this case a gradient is still possible to compute).
    redecode: bool
        Flag to indicate whether the probability distribution over the
        vocabulary should be recomputed after adjusting activations
        during interventions.
    diagnostic_classifiers: dict
        Dictionary of path to diagnostic classifiers to their respective
        diagnostic classifiers objects.
    intervention_points: list
        List of strings specifying on which layer for which activations
        interventions should be conducted, i.e. ['hx_l0', 'cx_l1']
    """

    def __init__(
        self,
        model: ForwardLSTM,
        diagnostic_classifiers: DiagnosticClassifierDict,
        intervention_points: List[str],
        step_size: float,
        trigger_func: Callable = None,
        masking: bool = False,
        redecode: bool = False,
    ):

        super().__init__(
            model,
            trigger_func=self.dc_trigger_func if trigger_func is None else trigger_func,
        )
        self.step_size = step_size
        self.masking = masking
        self.redecode = redecode

        # Link diagnostic classifiers to layer they correspond to
        self.diagnostic_classifiers = defaultdict(dict)

        for path, dc in diagnostic_classifiers.items():
            matches = re.search("(\wx)_(l\d+)", path)
            activation_type, layer = matches.groups()
            self.diagnostic_classifiers[layer][activation_type] = dc

        self.intervention_points = intervention_points
        self.topmost_layer = sorted(self.diagnostic_classifiers.keys())[-1]
        self.topmost_layer_num = int(self.topmost_layer[1:])

    @abstractmethod
    def select_diagnostic_classifier(
        self,
        inp: str,
        prev_activations: ActivationTensors,
        layer: str,
        activation_type: str,
        **additional: dict
    ) -> None:
        """
        Select the appropriate Diagnostic Classifier based on data used in current forward pass.

        Parameters
        ----------
        inp: str
            Current input token.
        prev_activations: FullActivationDict
            Activations of the previous time step.
        layer: str
            Identifier for current layer the intervention is being conducted on.
        activation_type: str
            Identifier for type of intervention the intervention is being conducted on.
        additional: dict
            Dictionary of additional information delivered via keyword arguments.
        """
        ...

    @abstractmethod
    def dc_trigger_func(
        self,
        prev_activations: ActivationTensors,
        activations: ActivationTensors,
        out: Tensor,
        prediction: Tensor,
        **additional: dict
    ) -> Tensor:
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

    def diagnostic_classifier_to_vars(
        self, diagnostic_classifier: LogReg
    ) -> Tuple[Tensor, Tensor]:
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
    def _wrap_in_var(array: np.array, requires_grad: bool) -> Variable:
        """
        Wrap a numpy array into a PyTorch Variable.

        Parameters
        ----------
        array: np.array
            Numpy array to be converted to a PyTorch Variable.
        requires_grad: bool
            Whether the variable requires the calculation of its gradients.
        """
        return Variable(
            torch.tensor(array, dtype=torch.float).squeeze(0),
            requires_grad=requires_grad,
        )

    @abstractmethod
    def diagnostic_classifier_loss(self, prediction: Tensor, label: Tensor) -> _Loss:
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
    def intervention_func(
        self,
        inp: str,
        prev_activations: ActivationTensors,
        out: Tensor,
        activations: ActivationTensors,
        **additional: Dict
    ) -> Tuple[Tensor, ActivationTensors]:
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
        for intervention_point in self.intervention_points:
            activation_type, layer = intervention_point.split("_")
            layer_num = int(layer[1:])
            dc = self.select_diagnostic_classifier(
                inp, prev_activations, layer, activation_type, **additional
            )
            weights, bias = self.diagnostic_classifier_to_vars(dc)
            label = torch.tensor([additional["label"]], dtype=torch.float)

            # Calculate gradient of the diagnostic classifier's prediction w.r.t.
            # the current activations
            current_activations = activations[layer_num][activation_type]
            current_activations = self._wrap_in_var(
                current_activations, requires_grad=True
            )
            params = [current_activations]
            optimizer = SGD(params, lr=self.step_size)
            optimizer.zero_grad()

            prediction = torch.sigmoid(weights @ current_activations + bias).unsqueeze(
                0
            )
            mask = self.dc_trigger_func(
                prev_activations, activations, out, prediction, **additional
            )
            loss = self.diagnostic_classifier_loss(prediction, label)
            loss.backward()
            gradient = current_activations.grad
            gradient = self.replace_nans(
                gradient
            )  # Sometimes gradient might become nan when loss is exactly 0

            # Manual (masked) update step
            new_activations = current_activations - self.step_size * gradient * mask
            activations[layer_num][activation_type] = new_activations

        # Repeat decoding step with adjusted activations
        if self.redecode:
            topmost_activations = activations[self.topmost_layer_num]["hx"]
            out: Tensor = self.model.w_decoder @ topmost_activations + self.model.b_decoder

        return out, activations

    @staticmethod
    def replace_nans(tensor: Tensor) -> Tensor:
        """ Replace nans in a PyTorch tensor with zeros. """
        tensor[tensor != tensor] = 0  # Exploit the fact that nan != nan

        return tensor
