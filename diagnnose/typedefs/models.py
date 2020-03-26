from typing import Union

from diagnnose.models.recurrent_lm import RecurrentLM
from diagnnose.models.transformer_lm import TransformerLM

LanguageModel = Union[RecurrentLM, TransformerLM]
