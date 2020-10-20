from diagnnose.config.config_dict import create_config_dict
from diagnnose.corpus import Corpus
from diagnnose.models import LanguageModel
from diagnnose.models.import_model import import_model
from diagnnose.probe.dc_trainer import DCTrainer
from diagnnose.tokenizer import create_tokenizer

if __name__ == "__main__":
    config_dict = create_config_dict()

    tokenizer = create_tokenizer(**config_dict["tokenizer"])
    corpus: Corpus = Corpus.create(tokenizer=tokenizer, **config_dict["corpus"])

    model: LanguageModel = import_model(config_dict)

    def selection_func(w_idx, item):
        return w_idx == len(item.sentence1) - 4

    dc_trainer = DCTrainer(
        **config_dict["init_dc"],
        corpus=corpus,
        model=model,
        train_selection_func=selection_func,
    )

    results = dc_trainer.train(**config_dict["train_dc"])
