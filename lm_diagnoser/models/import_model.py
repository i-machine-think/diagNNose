import json

from .forward_lstm import ForwardLSTM
from .language_model import LanguageModel


def import_model_from_json(model_file: str, vocab_file: str,
        module_location: str = '') -> LanguageModel:
    """
    Import a model from a json file.

    Arguments
    ----------
    model_file: str
        location of the pickled model file
    vocab_file: str
        location of the vocabulary of the model
    module_location: str, optional
        location of modules that should importable
        to load model from file

    Returns
    --------
    A LanguageModel from the given files
    """
    return ForwardLSTM(model_file, vocab_file, module_location)
