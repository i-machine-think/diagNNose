from typing import Type

import transformers

import diagnnose.models.wrappers as wrappers
from diagnnose.models import LanguageModel
from diagnnose.typedefs.config import ConfigDict
from diagnnose.vocab import get_vocab_path_from_config


def import_model(config_dict: ConfigDict) -> LanguageModel:
    """
    Import a model from a json file.

    Parameters
    ----------
    config_dict : ConfigDict
        Dictionary containing the model and init_states configuration.

    Returns
    --------
    model : LanguageModel
        A LanguageModel instance, based on the provided config_dict.
    """
    model_type = config_dict["model"].pop("type")

    if "transformers." in model_type:
        state_dict = config_dict["model"]["state_dict"]
        model_type = model_type.split(".")[1]

        model_constructor = getattr(transformers, model_type)
        model = model_constructor.from_pretrained(state_dict)

        tokenizer_constructor = getattr(transformers, config_dict["model"]["tokenizer"])
        model.tokenizer = tokenizer_constructor.from_pretrained(state_dict)
    else:
        model_constructor: Type[LanguageModel] = getattr(wrappers, model_type)
        model: LanguageModel = model_constructor(**config_dict["model"])

        vocab_path = get_vocab_path_from_config(config_dict)
        model.set_init_states(
            vocab_path=vocab_path, **config_dict.get("init_states", {})
        )

    config_dict["model"]["type"] = model_type

    return model
