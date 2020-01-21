import torch

from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import create_config_dict
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.decompositions.factory import DecomposerFactory
from diagnnose.models.import_model import import_model
from diagnnose.models.lm import LanguageModel
from diagnnose.typedefs.corpus import Corpus
from diagnnose.vocab import get_vocab_from_config

if __name__ == "__main__":
    arg_groups = {
        "model",
        "activations",
        "decompose",
        "init_states",
        "corpus",
        "init_states",
        "vocab",
    }
    arg_parser, required_args = create_arg_parser(arg_groups)
    config_dict = create_config_dict(arg_parser, required_args, arg_groups)

    vocab_path = get_vocab_from_config(config_dict)
    corpus: Corpus = import_corpus(vocab_path=vocab_path, **config_dict["corpus"])
    model: LanguageModel = import_model(config_dict)

    decompose_args = {
        **config_dict["decompose"],
        **config_dict["activations"],
    }

    print("Initializing decomposition")

    sen_ids = slice(0, 1)

    constructor = DecomposerFactory(model, corpus=corpus, sen_ids=sen_ids, **decompose_args)
    decomposer = constructor.create(sen_ids, classes=torch.tensor([0]), subsen_index=slice(0, None))

    print("Decomposing...")

    cd = decomposer.decompose(
        0, 1, ["rel-rel", "rel-b", "rel-irrel"], only_source_rel=True, decompose_o=True
    )
    print(cd)
