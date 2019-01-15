"""
Implementing a version of the intervention mechanism which relies on weak supervision, i.e. additional labels which
provide task-relevant information.
"""

from torch import Tensor

from interventions import InterventionMechanism
from models.forward_lstm import ForwardLSTM


class WeaklySupervisedInterventionMechanism(InterventionMechanism):
    """
    Mechanism that triggers an intervention based on additional labels and Diagnostic classifiers [1] [2].

    [1] https://www.jair.org/index.php/jair/article/view/11196/26408
    [2] https://arxiv.org/abs/1808.08079
    """
    def __init__(self,
                 model: ForwardLSTM,
                 ):
        pass

    def trigger_function(self) -> Tensor:
        pass


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