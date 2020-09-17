from typing import Optional

import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from torch import Tensor


class LogRegModule(nn.Module):
    def __init__(self, ninp: int, nout: int, rank: Optional[int] = None):
        super().__init__()

        if rank is None:
            self.classifier = nn.Linear(ninp, nout)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(ninp, rank), nn.Linear(rank, nout)
            )

    def forward(self, inp: Tensor, create_softmax=True):
        if create_softmax:
            return F.softmax(self.classifier(inp), dim=-1)
        return self.classifier(inp)


# https://github.com/skorch-dev/skorch/blob/master/docs/user/neuralnet.rst#subclassing-neuralnet
class L1NeuralNetClassifier(NeuralNetClassifier):
    def __init__(self, *args, lambda1=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda1 = lambda1

    def get_loss(self, y_pred, y_true, X=None, training=False):
        loss = super().get_loss(y_pred, y_true, X=X, training=training)
        loss += self.lambda1 * sum([w.abs().sum() for w in self.module_.parameters()])
        return loss
