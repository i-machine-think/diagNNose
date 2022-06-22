import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch import Tensor
from transformers import PreTrainedTokenizer

from diagnnose.activations.selection_funcs import final_token, only_mask_token
from diagnnose.corpus import Corpus
from diagnnose.extract import simple_extract
from diagnnose.models import LanguageModel
from diagnnose.typedefs.activations import SelectionFunc
from diagnnose.typedefs.syntax import AccuracyDict, ScoresDict, SyntaxEvalCorpora


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
        if isinstance(config.get("subtasks", None), str):
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

    def run(self) -> Tuple[AccuracyDict, ScoresDict]:
        """Performs the syntactic evaluation task that is initialised.

        Returns
        -------
        results : ResultsDict
            Dictionary mapping a task to a task condition to the model
            accuracy.
        """
        accuracies: AccuracyDict = {}
        scores: ScoresDict = {}

        for subtask, subtask_corpora in self.corpora.items():
            if isinstance(subtask_corpora, Corpus):
                scores_df = self._run_corpus(subtask_corpora)
                scores[subtask] = scores_df

                accuracy: float = (scores_df.scores > scores_df.counter_scores).mean()
                accuracies[subtask] = accuracy
            else:
                for condition, corpus in subtask_corpora.items():
                    scores_df = self._run_corpus(corpus)
                    scores.setdefault(subtask, {})[condition] = scores_df

                    accuracy: float = (
                        scores_df.scores > scores_df.counter_scores
                    ).mean()
                    accuracies.setdefault(subtask, {})[condition] = accuracy

        return accuracies, scores

    def _run_corpus(self, corpus: Corpus) -> pd.DataFrame:
        if self.model.is_causal:
            selection_func = final_token("sen")
        else:
            selection_func = only_mask_token(self.tokenizer.mask_token, "sen")

        if self.ignore_unk:
            sen_ids = self._create_non_unk_sen_ids(corpus)
            corpus = corpus.slice(sen_ids)
            if len(corpus) == 0:
                return pd.DataFrame(columns=["scores", "counter_scores"])

        activations = self._calc_final_hidden(corpus, selection_func)

        if "counter_sen" in corpus.fields:
            if self.model.is_causal:
                selection_func = final_token("counter_sen")
            else:
                selection_func = only_mask_token(
                    self.tokenizer.mask_token, "counter_sen"
                )

            corpus.sen_column = "counter_sen"
            counter_activations = self._calc_final_hidden(corpus, selection_func)
        else:
            counter_activations = None

        scores_df = self._calc_scores(
            corpus,
            activations,
            counter_activations=counter_activations,
        )

        return scores_df

    def _create_non_unk_sen_ids(self, corpus: Corpus) -> List[int]:
        """
        Creates a list of sen ids for which none of the items in that
        sentence are unknown to the tokenizer
        """
        sen_ids = []
        vocab = self.tokenizer.vocab

        # An unk token may neither appear in the prefix sen, nor be the eval token itself.
        for idx, ex in enumerate(corpus):
            if any(w not in vocab for w in ex.sen) or ex.token not in vocab:
                continue
            if hasattr(ex, "counter_token") and ex.counter_token not in vocab:
                continue
            if hasattr(ex, "counter_sen") and any(
                w not in vocab for w in ex.counter_sen
            ):
                continue

            sen_ids.append(idx)

        # skipped = len(corpus) - len(sen_ids)
        # if skipped:
        #     warnings.warn(f"{skipped} out of {len(corpus)} items skipped")

        return sen_ids

    def _calc_final_hidden(
        self,
        corpus: Corpus,
        selection_func: SelectionFunc,
    ) -> Tensor:
        activation_name = (self.model.top_layer, "hx")

        activation_reader, _ = simple_extract(
            self.model,
            corpus,
            [activation_name],
            batch_size=len(corpus),
            selection_func=selection_func,
        )

        activations = activation_reader.activation_dict[activation_name]

        return activations

    def _calc_scores(
        self,
        corpus: Corpus,
        activations: Tensor,
        counter_activations: Optional[Tensor] = None,
    ) -> pd.DataFrame:
        token_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(ex.token) for ex in corpus]
        )

        scores_df = pd.DataFrame(
            {
                "sen": [ex.sen for ex in corpus],
                "token": [ex.token for ex in corpus],
            }
        )

        if counter_activations is None:
            scores_df["counter_token"] = [ex.counter_token for ex in corpus]

            counter_token_ids = torch.tensor(
                [
                    self.tokenizer.convert_tokens_to_ids(ex.counter_token)
                    for ex in corpus
                ]
            )

            scores, counter_scores = self._single_context_accuracy(
                activations, token_ids, counter_token_ids
            )
        else:
            scores_df["counter_sen"] = [ex.counter_sen for ex in corpus]

            scores, counter_scores = self._dual_context_accuracy(
                activations, counter_activations, token_ids
            )

        scores_df["scores"] = scores.detach()
        scores_df["counter_scores"] = counter_scores.detach()

        return scores_df

    def _single_context_accuracy(
        self, activations: Tensor, token_ids: Tensor, counter_token_ids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """ Computes accuracy for comparing P(w1|h) > P(w2|h). """
        logits = self._decode(activations, token_ids)
        counter_logits = self._decode(activations, counter_token_ids)

        return logits, counter_logits

    def _dual_context_accuracy(
        self,
        activations: Tensor,
        counter_activations: Tensor,
        token_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """ Computes accuracy for comparing P(w|h1) > P(w|h2). """
        if self.use_full_model_probs:
            full_probs = self._decode(activations)
            counter_probs = self._decode(counter_activations)

            batch_size = full_probs.shape[0]
            probs = full_probs[range(batch_size), token_ids]
            counter_probs = counter_probs[range(batch_size), token_ids]
        else:
            probs = self._decode(activations, token_ids)
            counter_probs = self._decode(counter_activations, token_ids)

        return probs, counter_probs

    def _decode(
        self, activations: Tensor, token_ids: Optional[Tensor] = None
    ) -> Tensor:
        if hasattr(self.model, "decoder"):
            # Transformers
            logits = getattr(self.model, "decoder")(activations)

            if token_ids is not None:
                batch_size = logits.size(0)
                logits = logits[range(batch_size), token_ids]

            return logits
        elif hasattr(self.model, "decoder_w"):
            # LSTMs
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
        else:
            raise AttributeError("Model decoder not found")
