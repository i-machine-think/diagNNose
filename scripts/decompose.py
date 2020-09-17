import torch

from diagnnose.config.config_dict import create_config_dict
from diagnnose.corpus import Corpus
from diagnnose.decompositions.factory import DecomposerFactory
from diagnnose.models import LanguageModel
from diagnnose.models.import_model import import_model
from diagnnose.tokenizer.create import create_tokenizer

if __name__ == "__main__":
    config_dict = create_config_dict()

    model: LanguageModel = import_model(config_dict)
    tokenizer = create_tokenizer(**config_dict["tokenizer"])
    corpus: Corpus = Corpus.create(tokenizer=tokenizer, **config_dict["corpus"])

    decompose_args = {**config_dict["decompose"], **config_dict["activations"]}

    print("Initializing decomposition")

    sen_ids = slice(0, 10)

    factory = DecomposerFactory(model, corpus=corpus, sen_ids=sen_ids, **decompose_args)

    decomposer = factory.create(
        sen_ids, classes=torch.tensor([0]), subsen_index=slice(0, None)
    )

    print("Decomposing...")

    start_idx = 0
    end_idx = 1
    rel_interactions = ["rel-rel", "rel-b", "rel-irrel"]
    cd = decomposer.decompose(
        start_idx, end_idx, rel_interactions, only_source_rel=True
    )

    print(cd)
