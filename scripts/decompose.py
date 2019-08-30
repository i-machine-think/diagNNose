import torch

from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import create_config_dict
from diagnnose.decompositions.factory import DecomposerFactory
from diagnnose.models.import_model import import_model
from diagnnose.models.lm import LanguageModel

if __name__ == "__main__":
    arg_groups = {"model", "activations", "decompose", "init_states"}
    arg_parser, required_args = create_arg_parser(arg_groups)
    config_dict = create_config_dict(arg_parser, required_args, arg_groups)

    model: LanguageModel = import_model(config_dict)

    decompose_args = {**config_dict["decompose"], **config_dict["activations"]}

    constructor = DecomposerFactory(model, **decompose_args)
    decomposer = constructor.create([0, 1], slice(0, 4, 1), classes=torch.tensor([0]))

    cd = decomposer.decompose(
        0, 1, ["rel-rel", "rel-b", "rel-irrel"], only_source_rel=True, decompose_o=True
    )

    print(cd["relevant"].squeeze())
    print(cd["irrelevant"].squeeze())
