"""Model wrappers define the behaviour of a specific model architecture.

These wrapper classes should inherit from :class:`LanguageModel`, or one
of its subclasses, and adhere to the signature of the ``forward`` method
that is defined for it.
"""

from .awd_lstm import AWDLSTM
from .forward_lstm import ForwardLSTM
from .google_lm import GoogleLM
