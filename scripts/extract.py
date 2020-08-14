from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import create_config_dict
from diagnnose.corpus import Corpus
from diagnnose.extract import Extractor
from diagnnose.models import LanguageModel
from diagnnose.models.import_model import import_model
from diagnnose.tokenizer import create_tokenizer

if __name__ == "__main__":
    arg_groups = {"model", "activations", "corpus", "init_states"}
    arg_parser, required_args = create_arg_parser(arg_groups)
    config_dict = create_config_dict(arg_parser, required_args)

    tokenizer = create_tokenizer(config_dict["model"]["vocab"])
    corpus: Corpus = Corpus.create(**config_dict["corpus"])
    model: LanguageModel = import_model(config_dict)

    extractor = Extractor(model, corpus, **config_dict["activations"])
    a_reader = extractor.extract()
