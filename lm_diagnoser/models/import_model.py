import json

from models.forward_lstm import ForwardLSTM
from models.language_model import LanguageModel


def import_model_from_json(model_dir: str, model_class: ForwardLSTM=ForwardLSTM) -> LanguageModel:
    with open(f'{model_dir}/setup.json') as json_file:
        config = json.load(json_file)

    model_file = config['model_file']
    vocab_file = config['vocab_file']
    module_location = config['module_location']

    return model_class(model_file, vocab_file, module_location)
