from importlib import import_module
from typing import Type

from diagnnose.models.lm import LanguageModel
from diagnnose.typedefs.classifiers import LinearDecoder
from diagnnose.typedefs.config import ConfigDict
from diagnnose.vocab import get_vocab_from_config


def import_model(config_dict: ConfigDict) -> LanguageModel:
    """
    Import a model from a json file.

    Parameters
    ----------
    config_dict : ConfigDict
        Dictionary containing the model and init_states configuration.

    Returns
    --------
    A LanguageModel created from the given files
    """
    model_type = config_dict["model"].pop("type")

    module = import_module("diagnnose.model_wrappers")
    model_constructor: Type[LanguageModel] = getattr(module, model_type)
    model: LanguageModel = model_constructor(**config_dict["model"])

    vocab_path = get_vocab_from_config(config_dict)
    model.set_init_states(vocab_path=vocab_path, **config_dict["init_states"])

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
