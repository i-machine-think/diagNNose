from diagnnose.attribute import ContextualDecomposer, Explainer
from diagnnose.config import create_config_dict
from diagnnose.models import LanguageModel, import_model
from diagnnose.tokenizer import create_tokenizer
from diagnnose.utils.misc import profile

if __name__ == "__main__":
    config_dict = create_config_dict()

    model: LanguageModel = import_model(**config_dict["model"])
    tokenizer = create_tokenizer(**config_dict["tokenizer"])

    decomposer = ContextualDecomposer(model)
    explainer = Explainer(decomposer, tokenizer)

    sens = [f"The athletes above Barbara <mask>."]
    tokens = ["walk", "walks"]

    with profile():
        full_probs, contribution_probs = explainer.explain(sens, tokens)

    explainer.print_attributions(full_probs, contribution_probs, sens, tokens)
