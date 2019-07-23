from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import ConfigSetup
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.decompositions import DecomposerFactory
from diagnnose.models.import_model import import_model
from diagnnose.typedefs.models import LanguageModel
from diagnnose.typedefs.corpus import Corpus
from diagnnose.decompositions.attention import CDAttention


if __name__ == "__main__":
    arg_groups = {"model", "activations", "corpus", "vocab", "decompose"}
    arg_parser, required_args = create_arg_parser(arg_groups)
    print(1)
    config_dict = ConfigSetup(arg_parser, required_args, arg_groups).config_dict

    model: LanguageModel = import_model(config_dict["model"])
    corpus: Corpus = import_corpus(
        vocab_path=config_dict["vocab"]["vocab_path"], **config_dict["corpus"]
    )

    decompose_args = {**config_dict["decompose"], **config_dict["activations"]}

    factory = DecomposerFactory(model, **decompose_args)

    attention = CDAttention(factory)

    attention.plot_by_sen_id(0, corpus)
