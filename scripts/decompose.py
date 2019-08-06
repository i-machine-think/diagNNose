import torch

from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import ConfigSetup
from diagnnose.decompositions.factory import DecomposerFactory
from diagnnose.models.import_model import import_model
from diagnnose.models.lm import LanguageModel

if __name__ == "__main__":
    arg_groups = {"model", "activations", "decompose", "init_states"}
    arg_parser, required_args = create_arg_parser(arg_groups)

    config_dict = ConfigSetup(arg_parser, required_args, arg_groups).config_dict

    model: LanguageModel = import_model(
        config_dict["model"], config_dict["init_states"]
    )

    decompose_args = {**config_dict["decompose"], **config_dict["activations"]}

    constructor = DecomposerFactory(model, **decompose_args)
    decomposer = constructor.create(
        [0, 1], slice(0, 3, 1), classes=torch.tensor([[0], [1]])
    )

    cd = decomposer.decompose(-1, 0, ["rel-rel", "rel-b"])
    print(cd["relevant"])
    print(cd["irrelevant"])
