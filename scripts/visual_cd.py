from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import ConfigSetup
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.decompositions.attention import CDAttention
from diagnnose.models.import_model import import_model
from diagnnose.models.lm import LanguageModel
from diagnnose.typedefs.corpus import Corpus
from diagnnose.vocab import get_vocab_from_config

if __name__ == "__main__":
    arg_groups = {
        "model",
        "activations",
        "corpus",
        "init_states",
        "vocab",
        "decompose",
        "plot_attention",
    }
    arg_parser, required_args = create_arg_parser(arg_groups)

    config_dict = ConfigSetup(arg_parser, required_args, arg_groups).config_dict

    model: LanguageModel = import_model(
        config_dict["model"], config_dict["init_states"]
    )
    corpus: Corpus = import_corpus(
        vocab_path=get_vocab_from_config(config_dict), **config_dict["corpus"]
    )

    attention = CDAttention(
        model,
        corpus,
        cd_config=config_dict["decompose"],
        plot_config=config_dict["plot_attention"],
    )

    attention.plot_by_sen_id(
        slice(1200, 2400, 2), avg_decs=True, **config_dict["activations"]
    )
