from abc import ABC

from transformers import PreTrainedModel, PreTrainedTokenizer


class TransformerLM(ABC, PreTrainedModel):
    """ A TransformerLM is a HuggingFace model to which we attach
    a corresponding tokenizer.
    """
    tokenizer: PreTrainedTokenizer
