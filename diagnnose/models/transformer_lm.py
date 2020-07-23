from transformers import PreTrainedModel, PreTrainedTokenizer

from diagnnose.typedefs.models import LanguageModel


class TransformerLM(LanguageModel, PreTrainedModel):
    """ A TransformerLM is a HuggingFace model to which we attach
    a corresponding tokenizer.
    """

    tokenizer: PreTrainedTokenizer
