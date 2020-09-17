from typing import List, Tuple, Union

from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizer

from diagnnose.models import LanguageModel

from diagnnose.attribute.decomposer import ContextualDecomposer, ShapleyDecomposer

decomposers = {
    "shapley_decomposer": ShapleyDecomposer,
    "contextual_decomposer": ContextualDecomposer,
}


class Explainer:
    def __init__(
        self,
        model: LanguageModel,
        tokenizer: PreTrainedTokenizer,
        decomposer: str = "shapley_decomposer",
    ):
        self.decomposer = decomposers[decomposer](model)
        self.tokenizer = tokenizer

    def explain(self, input_tokens: Union[str, List[str]], output_tokens: List[str]):
        batch_encoding = self.tokenize(input_tokens)

        out, contributions = self.decomposer.decompose(batch_encoding)

        output_ids, mask_ids = self._create_output_ids(batch_encoding, output_tokens)

        full_probs = self._fetch_token_probs(out, output_ids, mask_ids)
        contribution_probs = [
            self._fetch_token_probs(contribution, output_ids, mask_ids)
            for contribution in contributions
        ]

        self._print_attributions(
            full_probs, contribution_probs, batch_encoding, output_tokens
        )

    def tokenize(self, sentences: Union[str, List[str]]) -> BatchEncoding:
        sentences = [sentences] if isinstance(sentences, str) else sentences

        batch_encoding = self.tokenizer(
            sentences,
            padding=True,
            return_attention_mask=False,
            return_length=True,
            return_token_type_ids=False,
        )

        return batch_encoding

    def _create_output_ids(
        self, batch_encoding: BatchEncoding, output_tokens: List[str]
    ) -> Tuple[List[int], List[int]]:
        [mask_idx, unk_idx] = self.tokenizer.convert_tokens_to_ids(["<mask>", "<unk>"])
        mask_idx = None if mask_idx == unk_idx else mask_idx
        mask_ids = [
            sen.index(mask_idx) if mask_idx in sen else final_idx - 1
            for sen, final_idx in zip(
                batch_encoding["input_ids"], batch_encoding["length"]
            )
        ]

        output_ids = []
        for token in output_tokens:
            # Simply encoding "token" often yields a different index than when it is embedded
            # within a sentence, hence the ugly hack here.
            token_id = self.tokenizer.encode(
                f"<mask> {token}", add_special_tokens=False
            )[1:]
            assert len(token_id) == 1, "Multi-subword tokens not supported"
            output_ids.append(token_id[0])

        return output_ids, mask_ids

    @staticmethod
    def _fetch_token_probs(
        probs: Tensor, output_ids: List[int], mask_ids: List[int]
    ) -> Tensor:
        """Fetches the probability of each output class at the position
        of the corresponding mask_idx.

        Parameters
        ----------
        probs : Tensor
            Tensor with output probabilities of shape:
            batch_size x max_sen_len x output_dim.
        output_ids : List[int]
            List of indices of the output classes that are decomposed.
        mask_ids : List[int]
            List of indices that signify the position of each sentence
            in the input batch where the decomposition will take place.

        Returns
        -------
        token_probs : Tensor
            Tensor containing the corresponding probabilities.
        """
        mask_probs = probs[range(probs.size(0)), mask_ids]
        token_probs = mask_probs[:, output_ids]

        return token_probs

    def _print_attributions(
        self,
        full_probs: Tensor,
        contribution_probs: List[Tensor],
        batch_encoding: BatchEncoding,
        output_tokens: List[str],
    ):
        for sen_idx, token_ids in enumerate(batch_encoding["input_ids"]):
            print((" " * 15) + "".join(f"{w:<15}" for w in output_tokens))
            print(
                f"{'Full logits':<15}"
                + "".join(f"{p:<15.3f}" for p in full_probs[sen_idx])
            )
            print("-" * 15 * (len(output_tokens) + 1))
            sen_features = [self.tokenizer.decode([w]) for w in token_ids]
            sen_len = batch_encoding["length"][sen_idx]
            features = ["model_bias", *sen_features[:sen_len]]
            for i, feature in enumerate(features):
                print(
                    f"{feature:<15}"
                    + "".join(f"{p:<15.3f}" for p in contribution_probs[i][sen_idx])
                )
            print("\n")
