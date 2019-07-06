from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import ConfigSetup
from diagnnose.models.import_model import import_model_from_json
from diagnnose.models.language_model import LanguageModel
from diagnnose.downstream.lakretz import lakretz_downstream

if __name__ == "__main__":
    arg_groups = {"model", "corpus", "downstream"}
    arg_parser, required_args = create_arg_parser(arg_groups)

    config_dict = ConfigSetup(arg_parser, required_args, arg_groups).config_dict

    model: LanguageModel = import_model_from_json(config_dict["model"])

    accs_dict = lakretz_downstream(
        model,
        config_dict["corpus"]["corpus_path"],
        config_dict["corpus"]["vocab_path"],
        lakretz_tasks=config_dict["downstream"]["lakretz_tasks"],
        device=config_dict["model"].get("device", "cpu"),
    )
