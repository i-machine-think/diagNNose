from __future__ import annotations
from typing import Type

from .language_model import LanguageModel


def import_model(*args, **kwargs) -> LanguageModel:
    """Import a model from a json file.

    Returns
    --------
    model : LanguageModel
        A LanguageModel instance, based on the provided config_dict.
    """

    if "transformer_type" in kwargs:
        model = _import_transformer_lm(*args, **kwargs)
    elif "rnn_type" in kwargs:
        model = _import_recurrent_lm(*args, **kwargs)
    else:
        raise TypeError("`transformer_type` or `rnn_type` must be provided as kwarg.")

    model.eval()

    device = kwargs.get("device", "cpu")
    model.device = device
    model.to(device)

    return model


def _import_transformer_lm(*args, **kwargs) -> "TransformerLM":
    """ Imports a Transformer LM. """
    from .transformer_lm import TransformerLM

    return TransformerLM(*args, **kwargs)


def _import_recurrent_lm(*args, **kwargs) -> "RecurrentLM":
    """ Imports a recurrent LM and sets the initial states. """
    from .recurrent_lm import RecurrentLM

    assert "rnn_type" in kwargs, "rnn_type should be given for recurrent LM"
    model_type = kwargs.pop("rnn_type")

    import diagnnose.models.wrappers as wrappers

    model_constructor: Type[RecurrentLM] = getattr(wrappers, model_type)
    model = model_constructor(*args, **kwargs)

    model.set_init_states()

    return model
