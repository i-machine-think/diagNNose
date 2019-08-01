from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import ConfigSetup
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.decompositions.attention import CDAttention
from diagnnose.models.import_model import import_model
from diagnnose.typedefs.corpus import Corpus
from diagnnose.models.lm import LanguageModel

if __name__ == "__main__":
    arg_groups = {"model", "activations", "corpus", "vocab", "decompose"}
    arg_parser, required_args = create_arg_parser(arg_groups)

    config_dict = ConfigSetup(arg_parser, required_args, arg_groups).config_dict

    model: LanguageModel = import_model({**config_dict["model"], **config_dict["vocab"]})
    corpus: Corpus = import_corpus(
        vocab_path=config_dict["vocab"]["vocab_path"], **config_dict["corpus"]
    )

    attention = CDAttention(model, cd_config=config_dict["decompose"])

    sen_id = 21

    attention.plot_by_sen_id(corpus, sen_id, **config_dict["activations"])
    attention.plot_by_sen_id(corpus, sen_id + 1, **config_dict["activations"])
