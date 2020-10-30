from diagnnose.attribute.decomposer import ShapleyDecomposer
from diagnnose.attribute.explainer import Explainer
from diagnnose.config import create_config_dict
from diagnnose.models import LanguageModel
from diagnnose.models.import_model import import_model
from diagnnose.tokenizer import create_tokenizer
from diagnnose.utils.misc import profile

if __name__ == "__main__":
    config_dict = create_config_dict()

    model: LanguageModel = import_model(config_dict)
    tokenizer = create_tokenizer(**config_dict["tokenizer"])

    decomposer = ShapleyDecomposer(model, num_samples=10)
    explainer = Explainer(decomposer, tokenizer)

    sens = [f"The author talked to Sara about {tokenizer.mask_token} book."]
    tokens = ["the", "their"]

    with profile():
        full_probs, contribution_probs = explainer.explain(sens, tokens)

    explainer.print_attributions(full_probs, contribution_probs, sens, tokens)
