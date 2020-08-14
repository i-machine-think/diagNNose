from pprint import pprint

from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import create_config_dict
from diagnnose.downstream.suite import DownstreamSuite
from diagnnose.models import LanguageModel
from diagnnose.models.import_model import import_model
from diagnnose.tokenizer import Tokenizer, create_tokenizer

if __name__ == "__main__":
    arg_groups = {"model", "vocab", "downstream", "init_states"}
    arg_parser, required_args = create_arg_parser(arg_groups)
    config_dict = create_config_dict(arg_parser, required_args)

    # Only pass along the selected set of tasks in the config
    if "tasks" in config_dict["downstream"]:
        config_dict["downstream"]["config"] = {
            task: config
            for task, config in config_dict["downstream"]["config"].items()
            if task in config_dict["downstream"]["tasks"]
        }

    model: LanguageModel = import_model(config_dict)
    tokenizer: Tokenizer = create_tokenizer(config_dict["tokenizer"])

    suite = DownstreamSuite(model, config_dict["downstream"]["config"], tokenizer)
    results = suite.run(use_full_model_probs=True)

    pprint(results)
