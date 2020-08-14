from .decomposer import Decomposer, ContextualDecomposer
from transformers import PreTrainedTokenizer
from diagnnose.models import LanguageModel
from typing import List, Tuple, Union
import torch
from diagnnose.attribute import ShapleyTensor
from torch import Tensor

decomposers = {"decomposer": Decomposer, "contextual_decomposer": ContextualDecomposer}


class Explainer:
    def __init__(
        self,
        model: LanguageModel,
        tokenizer: PreTrainedTokenizer,
        decomposer: str = "decomposer",
    ):
        self.decomposer = decomposers[decomposer](model)
        self.tokenizer = tokenizer

    def explain(self, *args, **kwargs):
        raise NotImplementedError

    def decompose(self, sentences: Union[str, List[str]]) -> ShapleyTensor:
        input_ids, attention_mask = self.tokenize(sentences)

        shapley_out = self.decomposer.decompose(input_ids, attention_mask)

        return shapley_out

    def tokenize(self, sentences: Union[str, List[str]]) -> Tuple[Tensor, Tensor]:
        sentences = [sentences] if isinstance(sentences, str) else sentences
        input_ids, attention_mask = self.tokenizer(sentences, padding=True).values()

        return torch.tensor(input_ids), torch.tensor(attention_mask)


class MaskedLMExplainer(Explainer):
    def explain(self, sentences: Union[str, List[str]], tokens: List[str]):
        out, contributions = self.decompose(sentences)

        input_ids = self.tokenize(sentences)[0].tolist()
        mask_idx = self.tokenizer.convert_tokens_to_ids("<mask>")
        mask_ids = [sen.index(mask_idx) for sen in input_ids]
        token_ids = self.create_token_ids(tokens)

        full_probs = self.fetch_token_probs(out, mask_ids, token_ids)
        contribution_probs = [
            self.fetch_token_probs(contribution, mask_ids, token_ids)
            for contribution in contributions
        ]

        self.print_attributions(full_probs, contribution_probs, input_ids, tokens)

    def print_attributions(
        self,
        full_probs: Tensor,
        contribution_probs: List[Tensor],
        input_ids: List[List[int]],
        tokens: List[str],
    ):
        for sen_idx in range(len(input_ids)):
            print((" " * 15) + "".join(f"{w:<15}" for w in tokens))
            print(
                f"{'Full logits':<15}"
                + "".join(f"{p:<15.3f}" for p in full_probs[sen_idx])
            )
            print("-" * 15 * (len(tokens) + 1))
            features = ["model_bias"]
            features += [
                self.tokenizer.decode(token_idx) for token_idx in input_ids[sen_idx]
            ]
            for i, feature in enumerate(features):
                print(
                    f"{feature:<15}"
                    + "".join(f"{p:<15.3f}" for p in contribution_probs[i][sen_idx])
                )
            print("\n")

    def create_token_ids(self, tokens: List[str]) -> List[int]:
        token_ids = []
        for token in tokens:
            token_id = self.tokenizer.encode(
                f"<mask> {token}", add_special_tokens=False
            )[1:]
            assert len(token_id) == 1, "Multi-subword tokens not supported"
            token_ids.append(token_id[0])

        return token_ids

    @staticmethod
    def fetch_token_probs(
        probs: Tensor, mask_ids: List[int], token_ids: List[int]
    ) -> Tensor:
        mask_probs = probs[range(probs.size(0)), mask_ids]
        token_probs = mask_probs[:, token_ids]

        return token_probs
