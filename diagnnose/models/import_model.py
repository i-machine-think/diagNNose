from typing import Type

from diagnnose.typedefs.config import ConfigDict

from .init_states import set_init_states
from .language_model import LanguageModel
from .recurrent_lm import RecurrentLM
from .transformer_lm import TransformerLM


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

    if "model_name" in config_dict["model"]:
        model = import_transformer_lm(config_dict)
    else:
        model = import_recurrent_lm(config_dict)

    model.eval()

    return model


def import_transformer_lm(config_dict: ConfigDict) -> TransformerLM:
    """ Imports a Transformer LM. """
    return TransformerLM(**config_dict["model"])


def import_recurrent_lm(config_dict: ConfigDict) -> RecurrentLM:
    """ Imports a recurrent LM and sets the initial states. """
    use_default = config_dict["model"].pop("use_default_init_states", False)
    model_type = config_dict["model"].pop("model_type")
    tokenizer = config_dict["model"].pop("tokenizer")

    import diagnnose.models.wrappers as wrappers

    model_constructor: Type[RecurrentLM] = getattr(wrappers, model_type)
    model = model_constructor(**config_dict["model"])

    set_init_states(
        model,
        use_default=use_default,
        tokenizer=tokenizer,
        **config_dict.get("init_states", {})
    )

    return model
