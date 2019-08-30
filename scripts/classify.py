from diagnnose.classifiers.dc_trainer import DCTrainer
from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import create_config_dict
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.typedefs.corpus import Corpus

if __name__ == "__main__":
    arg_groups = {"activations", "classify", "corpus"}
    arg_parser, required_args = create_arg_parser(arg_groups)
    config_dict = create_config_dict(arg_parser, required_args, arg_groups)

    corpus: Corpus = import_corpus(**config_dict["corpus"])

    dc_trainer = DCTrainer(
        **config_dict["activations"], **config_dict["classify"], corpus=corpus
    )

    dc_trainer.train()
