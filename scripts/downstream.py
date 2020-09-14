from pprint import pprint

from transformers import PreTrainedTokenizer

from diagnnose.config.config_dict import create_config_dict
from diagnnose.downstream.suite import DownstreamSuite
from diagnnose.models import LanguageModel
from diagnnose.models.import_model import import_model
from diagnnose.tokenizer import create_tokenizer

if __name__ == "__main__":
    arg_parser, required_args = create_arg_parser({"model", "downstream"})
    config_dict = create_config_dict(arg_parser, required_args)

    # Only pass along the selected set of tasks in the config
    if "tasks" in config_dict["downstream"]:
        config_dict["downstream"]["config"] = {
            task: config
            for task, config in config_dict["downstream"]["config"].items()
            if task in config_dict["downstream"]["tasks"]
        }

    model: LanguageModel = import_model(config_dict)
    tokenizer: PreTrainedTokenizer = create_tokenizer(**config_dict["tokenizer"])

    suite = DownstreamSuite(model, config_dict["downstream"]["config"], tokenizer)
    results = suite.run(use_full_model_probs=True)

    pprint(results)
