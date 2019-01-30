"""
Implementing a version of the intervention mechanism which relies on weak supervision, i.e. additional labels which
provide task-relevant information.
"""

from abc import abstractmethod, ABC
from typing import Dict, Tuple
import re

import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
from torch.nn import NLLLoss
from torch.nn.functional import sigmoid
from torch.optim import SGD
from overrides import overrides

from classifiers.diagnostic_classifier import DiagnosticClassifier
from interventions.mechanism import InterventionMechanism
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
        """
        ...

    def diagnostic_classifier_to_vars(self,
                                      diagnostic_classifier: DiagnosticClassifier) -> Tuple[Tensor, Tensor]:
        """
        Convert the weights and bias of a Diagnostic Classifier (trained with scikit-learn) into pytorch Variables.
        """
        weights = self._wrap_in_var(diagnostic_classifier.coef_, requires_grad=False)
        bias = self._wrap_in_var(diagnostic_classifier.intercept_, requires_grad=False)

        return weights, bias

    @staticmethod
    def _wrap_in_var(array: np.array,
                     requires_grad: bool) -> Variable:
        """
        Wrap a numpy array into a PyTorch Variable.
        """
        return Variable(torch.tensor(array, dtype=torch.float).squeeze(0), requires_grad=requires_grad)

    @abstractmethod
    def diagnostic_classifier_loss(self,
                                   prediction: Tensor,
                                   label: Tensor) -> _Loss:
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
    @overrides
    def select_diagnostic_classifier(self,
                                     inp: str,
                                     prev_activations: FullActivationDict,
                                     **additional: Dict):

        # Choose the classifier trained on topmost layer activations
        return self.diagnostic_classifiers[self.topmost_layer]

    @overrides
    def dc_trigger_func(self,
                        prev_activations: FullActivationDict,
                        activations: FullActivationDict,
                        out: Tensor,
                        prediction: Tensor,
                        **additional: dict) -> Tensor:
        """
        Trigger an intervention when the binary prediction for the sentence's number is incorrect.
        """
        label = additional["label"]
        wrong_predictions = torch.abs(prediction - label) >= 0.5
        wrong_predictions = wrong_predictions.float()

        return wrong_predictions

    @overrides
    def diagnostic_classifier_loss(self,
                                   prediction: Tensor,
                                   label: Tensor) -> _Loss:
        """
        Calculate the negative log-likelihood loss between the diagnostic classifiers prediction and the true class
        label by rephrasing the logistic regression into a 2-class multi-class classification problem.
        """
        class_predictions = torch.log(torch.cat((prediction, 1 - prediction))).unsqueeze(0)
        criterion = NLLLoss()
        loss = criterion(class_predictions, label.long())

        return loss


class SubjectLanguageModelInterventionMechanism(LanguageModelInterventionMechanism):
    """
    Like the Language Model Intervention Mechanism, except interventions are only possible at the subject's position.
    """

    @overrides
    def dc_trigger_func(self,
                        prev_activations: FullActivationDict,
                        activations: FullActivationDict,
                        out: Tensor,
                        prediction: Tensor,
                        **additional: dict) -> Tensor:
        """
        Trigger an intervention when the binary prediction for the sentence's number is incorrect.
        """
        label = additional["label"]
        is_subject_pos = additional["is_subj_pos"]
        mask = 0 if not is_subject_pos else 1
        wrong_predictions = torch.abs(prediction - label) >= 0.5
        wrong_predictions = wrong_predictions.float() * mask

        return wrong_predictions
