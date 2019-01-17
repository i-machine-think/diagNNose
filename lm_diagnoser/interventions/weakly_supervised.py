"""
Implementing a version of the intervention mechanism which relies on weak supervision, i.e. additional labels which
provide task-relevant information.
"""

from abc import abstractmethod, ABC
from typing import Dict, Tuple

import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
from torch.optim import SGD
from overrides import overrides

from classifiers import DiagnosticClassifier
from interventions import InterventionMechanism
from models.forward_lstm import ForwardLSTM
from typedefs.interventions import DiagnosticClassifierDict
from typedefs.models import FullActivationDict


class WeaklySupervisedInterventionMechanism(InterventionMechanism, ABC):
    """
    Mechanism that triggers an intervention based on additional labels and Diagnostic classifiers [1][2].

    [1] https://www.jair.org/index.php/jair/article/view/11196/26408
    [2] https://arxiv.org/abs/1808.08079
    """
    def __init__(self,
                 model: ForwardLSTM,
                 diagnostic_classifiers: DiagnosticClassifierDict,
                 step_size: float):

        super().__init__(model, trigger_func=self.dc_trigger_func)
        self.diagnostic_classifiers = diagnostic_classifiers
        self.step_size = step_size

    @abstractmethod
    def select_diagnostic_classifier(self,
                                     inp: str,
                                     prev_activations: FullActivationDict,
                                     **additional: Dict):
        """
        Select the appropriate Diagnostic Classifier based on data used in current forward pass.
        """
        ...

    @abstractmethod
    def dc_trigger_func(self,
                        out: Tensor,
                        diagnostic_classifier: DiagnosticClassifier,
                        label: str) -> Tensor:
        """
        Use a Diagnostic Classifier to determine for which batch instances to trigger an intervention.
        Returns a binary mask with 1 corresponding to an impending intervention.
        """
        ...

    def diagnostic_classifier_to_vars(self, diagnostic_classifier: DiagnosticClassifier) -> Tuple[Tensor, Tensor]:
        """
        Convert the weights and bias of a Diagnostic Classifier (trained with scikit-learn) into pytorch Variables.
        """
        weights = self._wrap_in_var(diagnostic_classifier.coef_, requires_grad=False)
        bias = self._wrap_in_var(diagnostic_classifier.intercept_, requires_grad=False)

        return weights, bias

    @staticmethod
    def _wrap_in_var(array, requires_grad):
        return Variable(Tensor(array, dtype=torch.double).squeeze(0), requires_grad=requires_grad)

    @abstractmethod
    def diagnostic_classifier_loss(self, prediction, label) -> _Loss:
        """
        Define in this function how the loss of the Diagnostic Classifier's prediction w.r.t to the loss is calculated.
        Should return a subclass of PyTorch's _Loss object like NLLLoss or CrossEntropyLoss.
        """
        ...

    @overrides
    def intervention_func(self,
                          inp: str,
                          prev_activations: FullActivationDict,
                          out: Tensor,
                          activations: FullActivationDict,
                          **additional: Dict) -> Tuple[Tensor, FullActivationDict]:

        dc = self.select_diagnostic_classifier(inp, prev_activations, **additional)
        weights, bias = self.diagnostic_classifier_to_vars(dc)
        label = additional["label"]

        # Create binary mask to determine for which activations in batch the intervention should be done
        mask = self.dc_trigger_func(out, dc, label)

        # Calculate gradient of the diagnostic classifier's prediction w.r.t. the current activations
        current_activations = self._wrap_in_var(out, requires_grad=True)
        params = [current_activations]
        optimizer = SGD(params, lr=self.step_size)
        optimizer.zero_grad()

        prediction = weights @ current_activations + bias
        loss = self.diagnostic_classifier_loss(prediction, label)
        loss.backward()

        # TODO: Do masking somewhere around here by performing optimizer step manually
        optimizer.step()

        # Convert back to normal tensor and return
        return Tensor(current_activations, dtype=torch.float), activations


class LanguageModelInterventionMechanism(WeaklySupervisedInterventionMechanism):
    """
    Intervention mechanism used in [1] for the Language Model used in [2].

    More specifically, a LSTM Language Model is trained and used to predict the probability of sentences in order to
    understand if LSTMs store information about subject and verb number. If so, the LM is expected to assign a higher
    probability to sentences where subject and verb are congruent.

    In the following, Diagnostic Classifiers [3] are used to predict the number of subject based on intermediate hidden
    states. If the prediction differs from the true label, the gradient of the prediction error w.r.t to the current
    activations is added to the activations themselves using the delta rule.

    [1] https://arxiv.org/abs/1808.08079
    [2] https://arxiv.org/abs/1803.11138
    [3] https://www.jair.org/index.php/jair/article/view/11196/26408
    """
    ...