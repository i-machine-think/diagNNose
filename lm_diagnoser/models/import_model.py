import json

from models.forward_lstm import ForwardLSTM
from models.language_model import LanguageModel


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
    module_location (optional): str
        location of modules that should importable
        to load model from file

    Returns
    --------
    A LanguageModel from the given files
    """
    # model_file = config['model_file']
    # vocab_file = config['vocab_file']
    # module_location = config['module_location']

    return ForwardLSTM(model_file, vocab_file, module_location)
