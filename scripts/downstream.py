from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import create_config_dict
from diagnnose.downstream.suite import DownstreamSuite
from diagnnose.models.import_model import import_model
from diagnnose.models.lm import LanguageModel
from diagnnose.vocab import get_vocab_from_config


if __name__ == "__main__":
    arg_groups = {"model", "vocab", "downstream", "init_states"}
    arg_parser, required_args = create_arg_parser(arg_groups)
    config_dict = create_config_dict(arg_parser, required_args, arg_groups)

    model: LanguageModel = import_model(config_dict)

    vocab_path = get_vocab_from_config(config_dict)
    assert vocab_path is not None, "vocab_path should be provided"

    suite = DownstreamSuite(
        config_dict["downstream"]["config"],
        vocab_path,
        device=config_dict["model"].get("device", "cpu"),
        print_results=True,
    )

    suite.run(model)
