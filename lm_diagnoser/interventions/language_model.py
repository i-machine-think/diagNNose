"""
Defining intervention mechanisms especially for language models.
"""

from typing import Dict

import torch
from overrides import overrides
from torch import Tensor
from torch.nn import NLLLoss
from torch.nn.modules.loss import _Loss

from ..interventions.weakly_supervised import WeaklySupervisedInterventionMechanism
from ..typedefs.models import FullActivationDict


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
        """
        Select the diagnostic classifier trained on the activations of the topmost layer.

        Parameters
        ----------
        inp: str
            Current input token.
        prev_activations: FullActivationDict
            Activations of the previous time step.
        additional: dict
            Dictionary of additional information delivered via keyword arguments.

        Returns
        -------
        diagnostic classifier: LogisticRegressionCV
            Selected diagnostic classifier.
        """
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

        Parameters
        ----------
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
        wrong_predictions: Tensor
            Binary mask indicating for which batch instances an intervention should be conducted.
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

        Parameters
        ----------
        prediction: Tensor
            Prediction of the diagnostic classifier based on the current time step's activations.
        label: Tensor
            Actual label to compare the prediction to.

        Returns
        -------
        loss: _Loss
            PyTorch loss between prediction and label.
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
        Trigger an intervention when the binary prediction for the sentence's number is incorrect, but only if it's also
        the time step corresponding to the sentence's subject.

        Parameters
        ----------
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
        wrong_predictions: Tensor
            Binary mask indicating for which batch instances an intervention should be conducted.
        """
        label = additional["label"]
        is_subject_pos = additional["is_subj_pos"]
        mask = 0 if not is_subject_pos else 1
        wrong_predictions = torch.abs(prediction - label) >= 0.5
        wrong_predictions = wrong_predictions.float() * mask

        return wrong_predictions
