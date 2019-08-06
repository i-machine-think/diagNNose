from importlib import import_module
from typing import Any, Dict, Type

from diagnnose.models.lm import LanguageModel
from diagnnose.typedefs.classifiers import LinearDecoder


def import_model(
    model_config: Dict[str, Any], init_states_config: Dict[str, Any]
) -> LanguageModel:
    """
    Import a model from a json file.

    Parameters
    ----------
    model_config : Dict[str, Any]
        Dictionary containing the model config attributes that are
        specific to that specific model.
    init_states_config : Dict[str, Any]
        Dictionary containing the init states config attributes.

    Returns
    --------
    A LanguageModel created from the given files
    """
    model_type = model_config.pop("model_type")

    module = import_module("diagnnose.model_wrappers")
    model_constructor: Type[LanguageModel] = getattr(module, model_type)
    model: LanguageModel = model_constructor(**model_config)

    model.set_init_states(**init_states_config)

    return model


def import_decoder_from_model(
    model: LanguageModel, decoder_w: str = "decoder_w", decoder_b: str = "decoder_b"
) -> LinearDecoder:
    """ Returns the decoding layer of a language model.

    Assumed to be a linear layer, that can be accessed by the decoder_w
    and decoder_b attributes of the model.

    Parameters
    ----------
    model : LanguageModel
        LanguageModel that contains a linear decoding layer.
    decoder_w : str
        Attribute name of the decoder coefficients in the LM.
    decoder_b : str
        Attribute name of the decoder bias in the LM.
    """
    w = getattr(model, decoder_w)
    b = getattr(model, decoder_b)

    return w, b
