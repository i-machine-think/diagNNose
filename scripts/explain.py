from diagnnose.attribute.explainer import Explainer
from diagnnose.config.config_dict import create_config_dict
from diagnnose.models import LanguageModel
from diagnnose.models.import_model import import_model
from diagnnose.tokenizer import create_tokenizer

if __name__ == "__main__":
    config_dict = create_config_dict()

    model: LanguageModel = import_model(config_dict)
    tokenizer = create_tokenizer(**config_dict["tokenizer"])

    explainer = Explainer(model, tokenizer, decomposer="shapley_decomposer")

    explainer.explain(["The athletes above Barbara <mask>."], ["approve", "approves"])
