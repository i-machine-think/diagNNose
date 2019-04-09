from typing import Type

from .forward_lstm import ForwardLSTM
from .language_model import LanguageModel


def import_model_from_json(model: str,
                           vocab: str,
                           lm_module: str,
                           device: str = 'cpu',
                           model_constructor: Type[LanguageModel] = ForwardLSTM) -> LanguageModel:
    """
    Import a model from a json file.

    Arguments
    ----------
    model : str
        Location of the pickled model file
    vocab : str
        Location of the vocabulary of the model
    lm_module : str, optional
        Location of modules that should imported to load model from file
    device : str
        Name of torch device on which model will be run. Defaults to cpu
    model_constructor: LanguageModel
        The class that the model is going to be loaded into.

    Returns
    --------
    A LanguageModel created from the given files
    """
    return model_constructor(model, vocab, lm_module, device)
