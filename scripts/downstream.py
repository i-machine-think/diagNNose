from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import ConfigSetup
from diagnnose.downstream.suite import DownstreamSuite
from diagnnose.models.import_model import import_model
from diagnnose.models.lm import LanguageModel
from diagnnose.vocab import get_vocab_from_config

if __name__ == "__main__":
    arg_groups = {"model", "vocab", "downstream", "init_states"}
    arg_parser, required_args = create_arg_parser(arg_groups)

    config_dict = ConfigSetup(arg_parser, required_args, arg_groups).config_dict

    model: LanguageModel = import_model(
        config_dict["model"], config_dict["init_states"]
    )

    suite = DownstreamSuite(
        device=config_dict["model"].get("device", "cpu"), **config_dict["downstream"]
    )

    vocab_path = get_vocab_from_config(config_dict)
    assert vocab_path is not None, "vocab_path should be provided"

    results = suite.perform_tasks(model, vocab_path)
