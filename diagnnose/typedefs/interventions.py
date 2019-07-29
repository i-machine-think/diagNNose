from typing import Any, Dict

from sklearn.linear_model import LogisticRegressionCV as LogReg

DiagnosticClassifierDict = Dict[Any, LogReg]
