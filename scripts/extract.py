from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import create_config_dict
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.extractors.base_extractor import Extractor
from diagnnose.models.import_model import import_model
from diagnnose.models.lm import LanguageModel
from diagnnose.typedefs.corpus import Corpus
from diagnnose.vocab import get_vocab_from_config

if __name__ == "__main__":
    arg_groups = {"model", "activations", "corpus", "extract", "init_states", "vocab"}
    arg_parser, required_args = create_arg_parser(arg_groups)
    config_dict = create_config_dict(arg_parser, required_args, arg_groups)

    model: LanguageModel = import_model(config_dict)
    corpus: Corpus = import_corpus(
        vocab_path=get_vocab_from_config(config_dict), **config_dict["corpus"]
    )

    extractor = Extractor(model, corpus, **config_dict["activations"])
    extractor.extract(**config_dict["extract"])
