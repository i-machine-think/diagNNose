from pprint import pprint

from transformers import PreTrainedTokenizer

from diagnnose.config.config_dict import create_config_dict
from diagnnose.models import LanguageModel, import_model, set_init_states
from diagnnose.syntax.evaluator import SyntacticEvaluator
from diagnnose.tokenizer import create_tokenizer
from diagnnose.utils.misc import profile

if __name__ == "__main__":
    config_dict = create_config_dict()

    model: LanguageModel = import_model(**config_dict["model"])
    tokenizer: PreTrainedTokenizer = create_tokenizer(**config_dict["tokenizer"])
    set_init_states(model, use_default=True, tokenizer=tokenizer)

    suite = SyntacticEvaluator(model, tokenizer, **config_dict["downstream"])
    accuracies, scores = suite.run()

    pprint(scores)
