from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import create_config_dict
from diagnnose.attribute import MaskedLMExplainer
from diagnnose.models import LanguageModel
from diagnnose.models.import_model import import_model
from diagnnose.tokenizer import create_tokenizer

if __name__ == "__main__":
    arg_parser, required_args = create_arg_parser({"model"})
    config_dict = create_config_dict(arg_parser, required_args)

    model: LanguageModel = import_model(config_dict)
    tokenizer = create_tokenizer(config_dict["tokenizer"])

    explainer = MaskedLMExplainer(model, tokenizer, decomposer="decomposer")

    explainer.explain(["The boy <mask>.", "The boys <mask>."], ["walk", "walks"])
