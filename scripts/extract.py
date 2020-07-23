from diagnnose.activations.selection_funcs import return_all
from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import create_config_dict
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.extract import Extractor
from diagnnose.models.import_model import import_model
from diagnnose.corpus import Corpus
from diagnnose.typedefs.models import LanguageModel
from diagnnose.vocab import get_vocab_path_from_config

if __name__ == "__main__":
    arg_groups = {"model", "activations", "corpus", "extract", "init_states", "vocab"}
    arg_parser, required_args = create_arg_parser(arg_groups)
    config_dict = create_config_dict(arg_parser, required_args)

    model: LanguageModel = import_model(config_dict)
    vocab_path = get_vocab_path_from_config(config_dict)
    corpus: Corpus = import_corpus(vocab_path=vocab_path, **config_dict["corpus"])

    extractor = Extractor(model, corpus, **config_dict["activations"], selection_func=return_all)
    a_reader = extractor.extract()
