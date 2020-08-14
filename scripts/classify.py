from typing import Optional

from torchtext.data import Example

from diagnnose.classifiers.dc_trainer import DCTrainer
from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import create_config_dict
from diagnnose.corpus import Corpus
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.models import LanguageModel
from diagnnose.models.import_model import import_model
from diagnnose.tokenizer import get_vocab_path_from_config

if __name__ == "__main__":
    arg_groups = {
        "activations",
        "init_dc",
        "train_dc",
        "corpus",
        "model",
        "vocab",
        "init_states",
    }
    arg_parser, required_args = create_arg_parser(arg_groups)
    config_dict = create_config_dict(arg_parser, required_args, validate=False)

    vocab_path = get_vocab_path_from_config(config_dict)

    corpus: Corpus = import_corpus(vocab_path=vocab_path, **config_dict["corpus"])
    test_corpus: Optional[Corpus] = None

    model: Optional[LanguageModel] = None
    if "model" in config_dict:
        model = import_model(config_dict)

    if config_dict["init_dc"].get("test_corpus", None) is not None:
        test_corpus = import_corpus(
            config_dict["init_dc"].pop("test_corpus"), vocab_path=vocab_path
        )

    label_vocab = corpus.fields["labels"].tokenizer.itos
    corpus_vocab = corpus.fields["sen"].tokenizer.stoi

    # Example selection_func/test_selection_func setup:
    # Evaluate on second hidden state, train on the rest
    def selection_func(_sen_id: int, pos: int, _item: Example):
        return pos != 1

    def test_selection_func(_sen_id: int, pos: int, _item: Example):
        return pos == 1

    # Use a random mapping based on the current token for the control task.
    def control_task(_sen_id: int, pos: int, item: Example):
        return corpus_vocab[item.sen[pos]] % len(label_vocab)

    dc_trainer = DCTrainer(
        **config_dict["activations"],
        **config_dict["init_dc"],
        corpus=corpus,
        test_corpus=test_corpus,
        model=model,
        selection_func=selection_func,
        test_selection_func=test_selection_func,
        control_task=control_task,
    )
    results = dc_trainer.train(**config_dict["train_dc"])
