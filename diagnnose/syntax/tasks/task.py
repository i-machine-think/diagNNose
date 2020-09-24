import glob
import os
import warnings
from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from torchtext.data import Example
from transformers import PreTrainedTokenizer

from diagnnose.activations.selection_funcs import final_token
from diagnnose.corpus import Corpus
from diagnnose.extract import simple_extract
from diagnnose.models import LanguageModel
from diagnnose.typedefs.activations import SelectionFunc

# subtask -> Corpus | (condition -> Corpus)
SyntaxEvalCorpora = Dict[str, Union[Corpus, Dict[str, Corpus]]]
# subtask -> accuracy | (condition -> accuracy)
ResultsDict = Dict[str, Union[float, Dict[str, float]]]


class SyntaxEvalTask:
    """Base class for syntactic evaluation tasks, from which specific
    tasks can inherit.

    Parameters
    ----------
    model : LanguageModel
        Language model for which the accuracy is calculated.
    tokenizer : PreTrainedTokenizer
        The model tokenizer that converts tokens into indices.
    config : Dict[str, Any]
        Configuration dictionary containing the setup for task
        initialization.
    use_full_model_probs : bool, optional
        Toggle to calculate the full model probs for the NPI
        sentences. If set to False only the NPI logits will be
        compared, instead of their Softmax probabilities. Defaults
        to True.
    ignore_unk : bool, optional
        Ignore cases for which at least one of the cases of the verb
        is not part of the model's tokenizer. Defaults to False.
    """

    def __init__(
        self,
        model: LanguageModel,
        tokenizer: PreTrainedTokenizer,
        ignore_unk: bool,
        use_full_model_probs: bool,
        **config: Dict[str, Any],
    ):
        model.eval()
        self.model = model
        self.tokenizer = tokenizer

        self.ignore_unk = ignore_unk
        self.use_full_model_probs = use_full_model_probs

        # If a single subtask is passed as cmd arg it is not converted to a list yet
        if "subtasks" in config and isinstance(config["subtasks"], str):
            config["subtasks"] = [config["subtasks"]]

        self.corpora: SyntaxEvalCorpora = self.initialize(**config)

    def initialize(
        self, path: str, header: Optional[List[str]] = None
    ) -> SyntaxEvalCorpora:
        if header is None:
            header = ["sen", "token", "counter_token"]

        assert "sen" in header
        assert "token" in header
        assert "counter_sen" in header or "counter_token" in header

        corpora = {}

        if os.path.isdir(path):
            for file in glob.glob(os.path.join(path, "*")):
                corpus = Corpus.create(file, header=header, tokenizer=self.tokenizer)
                task_name = file.split("/")[-1].split(".")[0]
                corpora[task_name] = corpus
        elif os.path.isfile(path):
            corpus = Corpus.create(path, header=header, tokenizer=self.tokenizer)
            task_name = path.split("/")[-1].split(".")[0]
            corpora[task_name] = corpus
        else:
            raise FileNotFoundError("Path to task is not found")

        return corpora

    def run(self) -> ResultsDict:
        """Performs the syntactic evaluation task that is initialised.

        Returns
        -------
        results : ResultsDict
            Dictionary mapping a task to a task condition to the model
            accuracy.
        """
        results: ResultsDict = {}

        for subtask, subtask_corpora in self.corpora.items():
            if isinstance(subtask_corpora, Corpus):
                accuracy = self.run_corpus(subtask_corpora)
                results[subtask] = accuracy
            else:
                for condition, corpus in subtask_corpora.items():
                    accuracy = self.run_corpus(corpus)

                    results.setdefault(subtask, {})[condition] = accuracy

        return results

    def run_corpus(self, corpus: Corpus) -> float:
        if self.tokenizer.mask_token is not None:
            def selection_func(w_idx: int, item: Example) -> bool:
                return item.sen[w_idx] == self.tokenizer.mask_token
        else:
            selection_func = final_token

        activations = self.calc_final_hidden(corpus, selection_func)
        counter_activations = None

        if "counter_sen" in corpus.fields:
            def selection_func(w_idx: int, item: Example) -> bool:
                if self.tokenizer.mask_token is not None:
                    return item.counter_sen[w_idx] == self.tokenizer.mask_token
                else:
                    return len(item.counter_sen) == (w_idx + 1)

            counter_activations = self.calc_final_hidden(
                corpus, selection_func, sen_column="counter_sen"
            )

        accuracy = self.calc_accuracy(
            corpus,
            activations,
            counter_activations=counter_activations,
        )

        return accuracy

    def calc_final_hidden(
        self,
        corpus: Corpus,
        selection_func: SelectionFunc,
        sen_column: str = "sen",
    ) -> Tensor:
        activation_name = (self.model.top_layer, "hx")

        activation_reader, _ = simple_extract(
            self.model,
            corpus,
            [activation_name],
            batch_size=len(corpus),
            selection_func=selection_func,
            sen_column=sen_column,
        )

        activations = torch.cat(activation_reader[:, activation_name], dim=0)

        return activations

    def calc_accuracy(
        self,
        corpus: Corpus,
        activations: Tensor,
        counter_activations: Optional[Tensor] = None,
    ) -> float:
        mask = self.create_unk_sen_mask(corpus)

        activations = activations[mask]

        token_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(ex.token) for ex in corpus]
        )
        token_ids = token_ids[mask]

        if counter_activations is None:
            counter_token_ids = torch.tensor(
                [
                    self.tokenizer.convert_tokens_to_ids(ex.counter_token)
                    for ex in corpus
                ]
            )
            counter_token_ids = counter_token_ids[mask]

            accuracy = self.single_context_accuracy(
                activations, token_ids, counter_token_ids
            )
        else:
            accuracy = self.dual_context_accuracy(
                activations, counter_activations[mask], token_ids
            )

        return accuracy

    def create_unk_sen_mask(self, corpus: Corpus) -> Tensor:
        """
        Creates a tensor mask for sentences that contain at least one
        token that is not part of the model's tokenizer.
        """
        mask = torch.ones(len(corpus), dtype=torch.bool)
        if not self.ignore_unk:
            return mask

        for idx, ex in enumerate(corpus):
            for w in ex.sen:
                if w not in self.tokenizer.vocab:
                    mask[idx] = False
                    warnings.warn(f"'{w}' is not part of model's tokenizer!")

        return mask

    def single_context_accuracy(
        self, activations: Tensor, token_ids: Tensor, counter_token_ids: Tensor
    ) -> float:
        """ Computes accuracy for comparing P(w1|h) > P(w2|h). """
        logits = self.decode(activations, token_ids)
        counter_logits = self.decode(activations, counter_token_ids)

        return torch.mean((logits >= counter_logits).to(torch.float)).item()

    def dual_context_accuracy(
        self,
        activations: Tensor,
        counter_activations: Tensor,
        token_ids: Tensor,
    ) -> float:
        """ Computes accuracy for comparing P(w|h1) > P(w|h2). """
        if self.use_full_model_probs:
            full_probs = self.decode(activations)
            counter_probs = self.decode(counter_activations)

            batch_size = full_probs.shape[0]
            probs = full_probs[range(batch_size), token_ids]
            counter_probs = counter_probs[range(batch_size), token_ids]
        else:
            probs = self.decode(activations, token_ids)
            counter_probs = self.decode(counter_activations, token_ids)

        return torch.mean((probs >= counter_probs).to(torch.float)).item()

    def decode(self, activations: Tensor, token_ids: Optional[Tensor] = None) -> Tensor:
        if hasattr(self.model, "decoder_w"):
            decoder_w = self.model.decoder_w
            decoder_b = self.model.decoder_b
            if token_ids is None:
                logits = activations @ decoder_w.t() + decoder_b

                return torch.nn.functional.log_softmax(logits, dim=-1)
            else:
                decoder_w = decoder_w[token_ids].unsqueeze(1)

                logits = torch.bmm(decoder_w, activations.unsqueeze(2)).squeeze()
                logits += decoder_b[token_ids]

                return logits
        elif hasattr(self.model, "decoder"):
            logits = getattr(self.model, "decoder")(activations)

            if token_ids is not None:
                batch_size = logits.size(0)
                logits = logits[range(batch_size), token_ids]

            return logits
        else:
            raise AttributeError("Model decoder not found")
