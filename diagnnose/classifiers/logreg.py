import torch.nn as nn
from typing import Optional
from torch import Tensor
import torch.nn.functional as F


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
