from pprint import pprint

from transformers import PreTrainedTokenizer

from diagnnose.config.config_dict import create_config_dict
from diagnnose.syntax.evaluator import SyntacticEvaluator
from diagnnose.models import LanguageModel
from diagnnose.models.import_model import import_model
from diagnnose.tokenizer import create_tokenizer

if __name__ == "__main__":
    config_dict = create_config_dict()

    model: LanguageModel = import_model(config_dict)
    tokenizer: PreTrainedTokenizer = create_tokenizer(**config_dict["tokenizer"])

    suite = SyntacticEvaluator(model, tokenizer, **config_dict["downstream"])
    results = suite.run()

    pprint(results)
