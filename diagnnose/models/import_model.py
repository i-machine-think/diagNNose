from typing import Type

from .init_states import _create_zero_states
from .language_model import LanguageModel
from .recurrent_lm import RecurrentLM
from .transformer_lm import TransformerLM


def import_model(*args, **kwargs) -> LanguageModel:
    """Import a model from a json file.

    Returns
    --------
    model : LanguageModel
        A LanguageModel instance, based on the provided config_dict.
    """

    if "model_name" in kwargs:
        model = _import_transformer_lm(*args, **kwargs)
    else:
        model = _import_recurrent_lm(*args, **kwargs)

    model.eval()

    return model


def _import_transformer_lm(*args, **kwargs) -> TransformerLM:
    """ Imports a Transformer LM. """
    return TransformerLM(*args, **kwargs)


def _import_recurrent_lm(*args, **kwargs) -> RecurrentLM:
    """ Imports a recurrent LM and sets the initial states. """

    assert "model_type" in kwargs, "model_type should be given for recurrent LM"
    model_type = kwargs.pop("model_type")

    import diagnnose.models.wrappers as wrappers

    model_constructor: Type[RecurrentLM] = getattr(wrappers, model_type)
    model = model_constructor(*args, **kwargs)

    model.init_states = _create_zero_states(model)

    return model
