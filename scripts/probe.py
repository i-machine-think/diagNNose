from diagnnose.activations.selection_funcs import final_sen_token, intersection
from diagnnose.config import create_config_dict
from diagnnose.corpus import Corpus
from diagnnose.models import LanguageModel, import_model, set_init_states
from diagnnose.probe import DataLoader, DCTrainer
from diagnnose.tokenizer import create_tokenizer

if __name__ == "__main__":
    config_dict = create_config_dict()

    tokenizer = create_tokenizer(**config_dict["tokenizer"])
    corpus: Corpus = Corpus.create(tokenizer=tokenizer, **config_dict["corpus"])
    model: LanguageModel = import_model(**config_dict["model"])
    set_init_states(model, use_default=True, tokenizer=tokenizer)

    def selection_func(_, item) -> bool:
        return item.env == "adverbs"

    data_loader = DataLoader(
        corpus,
        model=model,
        train_test_ratio=0.9,
        activation_names=[(1, "hx")],
        train_selection_func=intersection((selection_func, final_sen_token)),
    )

    dc_trainer = DCTrainer(
        data_loader,
        **config_dict["probe"],
    )

    results = dc_trainer.train()
