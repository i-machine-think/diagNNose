from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import ConfigSetup
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.extractors.base_extractor import Extractor
from diagnnose.models.import_model import import_model
from diagnnose.typedefs.corpus import Corpus
from diagnnose.models.lm import LanguageModel

if __name__ == "__main__":
    arg_groups = {"model", "activations", "corpus", "vocab", "extract"}
    arg_parser, required_args = create_arg_parser(arg_groups)

    config_dict = ConfigSetup(arg_parser, required_args, arg_groups).config_dict

    model: LanguageModel = import_model({**config_dict["model"], **config_dict["vocab"]})
    corpus: Corpus = import_corpus(
        vocab_path=config_dict["vocab"]["vocab_path"], **config_dict["corpus"]
    )

    extractor = Extractor(model, corpus, **config_dict["activations"])
    extractor.extract(**config_dict["extract"])
