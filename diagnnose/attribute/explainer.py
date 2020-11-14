from typing import List, Tuple, Union

from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizer

from .decomposer import Decomposer


class Explainer:
    """ Generates an explanation for a specific input. """

    def __init__(
        self,
        decomposer: Decomposer,
        tokenizer: PreTrainedTokenizer,
    ):
        self.decomposer = decomposer
        self.tokenizer = tokenizer

    def explain(self, input_tokens: Union[str, List[str]], output_tokens: List[str]):
        batch_encoding = self._tokenize(input_tokens)

        out, contributions = self.decomposer.decompose(batch_encoding)

        output_ids, mask_ids = self._create_output_ids(batch_encoding, output_tokens)

        full_probs = self._fetch_token_probs(out, output_ids, mask_ids)
        contribution_probs = [
            self._fetch_token_probs(contribution, output_ids, mask_ids)
            for contribution in contributions
        ]

        return full_probs, contribution_probs

    def _tokenize(self, input_tokens: Union[str, List[str]]) -> BatchEncoding:
        input_tokens = [input_tokens] if isinstance(input_tokens, str) else input_tokens

        batch_encoding = self.tokenizer(
            input_tokens,
            padding=True,
            return_attention_mask=False,
            return_length=True,
            return_token_type_ids=False,
        )

        return batch_encoding

    def _create_output_ids(
        self, batch_encoding: BatchEncoding, output_tokens: List[str]
    ) -> Tuple[List[int], List[int]]:
        mask_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        unk_idx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)
        mask_idx = None if mask_idx == unk_idx else mask_idx
        mask_ids = [
            sen.index(mask_idx) if mask_idx in sen else final_idx - 1
            for sen, final_idx in zip(
                batch_encoding["input_ids"], batch_encoding["length"]
            )
        ]

        output_ids = []
        mask_token = self.tokenizer.mask_token or "<mask>"
        for token in output_tokens:
            sub_token_id = self.tokenizer.convert_tokens_to_ids(token)

            if sub_token_id == unk_idx:
                # Simply encoding "token" often yields a different index than when it is embedded
                # within a sentence, hence the ugly hack here.
                sub_token_ids = self.tokenizer.encode(
                    f"{mask_token} {token}", add_special_tokens=False
                )[1:]
                sub_tokens = self.tokenizer.convert_ids_to_tokens(sub_token_ids)
                assert (
                    len(sub_token_ids) == 1
                ), f"Multi-subword tokens not supported ({token} -> {str(sub_tokens)})"
                sub_token_id = sub_token_ids[0]

            output_ids.append(sub_token_id)

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

    def print_attributions(
        self,
        full_probs: Tensor,
        contribution_probs: List[Tensor],
        input_tokens: Union[str, List[str]],
        output_tokens: List[str],
    ):
        batch_encoding = self._tokenize(input_tokens)

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
