from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import ConfigSetup
from diagnnose.decompositions.factory import DecomposerFactory
from diagnnose.models.import_model import import_model_from_json

if __name__ == "__main__":
    arg_groups = {"model", "activations", "decompose"}
    arg_parser, required_args = create_arg_parser(arg_groups)

    config_dict = ConfigSetup(arg_parser, required_args, arg_groups).config_dict

    model = import_model_from_json(config_dict["model"])

    decompose_args = {**config_dict["decompose"], **config_dict["activations"]}

    constructor = DecomposerFactory(model, **decompose_args)
    decomposer = constructor.create(0, slice(0, 6, 1))

    cd = decomposer.decompose(1, 2, ["rel-rel", "rel-b"])
    print(cd["relevant"], cd["irrelevant"])
