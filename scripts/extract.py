from diagnnose.config.config_dict import create_config_dict
from diagnnose.corpus import Corpus
from diagnnose.extract import Extractor
from diagnnose.models import LanguageModel, import_model, set_init_states
from diagnnose.tokenizer.create import create_tokenizer

if __name__ == "__main__":
    config_dict = create_config_dict()

    tokenizer = create_tokenizer(**config_dict["tokenizer"])
    corpus: Corpus = Corpus.create(tokenizer=tokenizer, **config_dict["corpus"])
    model: LanguageModel = import_model(**config_dict["model"])
    set_init_states(model, tokenizer=tokenizer, use_default=True)

    extractor = Extractor(model, corpus, **config_dict["extract"])
    a_reader = extractor.extract()
