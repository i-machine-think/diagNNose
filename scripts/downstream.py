from pprint import pprint

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

    vocab_path = get_vocab_from_config(config_dict)
    assert vocab_path is not None, "vocab_path should be provided"

    # Only pass along the selected set of tasks in the config
    if "tasks" in config_dict["downstream"]:
        config_dict["downstream"]["config"] = {
            task: config
            for task, config in config_dict["downstream"]["config"].items()
            if task in config_dict["downstream"]["tasks"]
        }

    suite = DownstreamSuite(
        config_dict["downstream"]["config"],
        vocab_path,
        device=config_dict["model"].get("device", "cpu"),
        print_results=True,
    )

    model: LanguageModel = import_model(config_dict)
    results = suite.run(model, ignore_unk=True)

    pprint(results)
