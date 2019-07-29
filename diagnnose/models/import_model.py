from importlib import import_module
from typing import Any, Dict, Type

from diagnnose.typedefs.classifiers import LinearDecoder
from diagnnose.typedefs.lm import LanguageModel


def import_model(model_config: Dict[str, Any]) -> LanguageModel:
    """
    Import a model from a json file.

    Arguments
    ----------
    model_config : str
        Dictionary containing the model config attributes that are
        specific to that specific model.

    Returns
    --------
    A LanguageModel created from the given files
    """
    model_type = model_config.pop("model_type")

    module = import_module("diagnnose.models")
    model_constructor: Type[LanguageModel] = getattr(module, model_type)

    return model_constructor(**model_config)


def import_decoder_from_model(
    model: LanguageModel, decoder_w: str = "decoder_w", decoder_b: str = "decoder_b"
) -> LinearDecoder:
    """ Returns the decoding layer of a language model.

    Assumed to be a linear layer, that can be accessed by the decoder_w
    and decoder_b attributes of the model.

    Arguments
    ---------
    model : LanguageModel
        LanguageModel that contains a linear decoding layer.
    decoder_w : str
        Attribute name of the decoder coefficients in the LM.
    decoder_b : str
        Attribute name of the decoder bias in the LM.
    """
    w = getattr(model, decoder_w)
    b = getattr(model, decoder_b)

    if model.array_type == "torch":
        w = w.data.numpy()
        if b is not None:
            b = b.data.numpy()

    return w, b
