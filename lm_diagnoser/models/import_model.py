from models.language_model import LanguageModel
from models.forward_lstm import ForwardLSTM

import json


def import_model_from_json(filename: str) -> LanguageModel:
    with open(filename) as json_file:
        config = json.load(json_file)

    model_file = config['model_file']
    vocab_file = config['vocab_file']
    module_location = config['module_location']

    return ForwardLSTM(model_file, vocab_file, module_location)
