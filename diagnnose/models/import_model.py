# TODO: Removing this v somehow breaks RecurrentLM import in ForwardLSTM?
import diagnnose.models.wrappers as wrappers
from diagnnose.models import LanguageModel
from diagnnose.tokenizer import create_tokenizer
from diagnnose.typedefs.config import ConfigDict

from .rnn import RecurrentLM
from .transformer import TransformerLM


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
        model = TransformerLM(**config_dict["model"])
    else:
        use_default_init_states = config_dict["model"].pop(
            "use_default_init_states", False
        )
        tokenizer = create_tokenizer(config_dict["tokenizer"])

        model = RecurrentLM.create_from_type(**config_dict["model"])
        model.set_init_states(
            use_default=use_default_init_states,
            tokenizer=tokenizer,
            **config_dict.get("init_states", {})
        )

    model.eval()

    return model
