import os
from typing import Dict, List, Optional

from torchtext.data import RawField

from diagnnose.corpus import Corpus
from diagnnose.typedefs.models import LanguageModel

from ..task import DownstreamTask, DownstreamCorpora


class LakretzDownstream(DownstreamTask):
    """

    Parameters
    ----------
    """

    descriptions = {
        "adv": {"items_per_condition": 900, "conditions": ["S", "P"]},
        "adv_adv": {"items_per_condition": 900, "conditions": ["S", "P"]},
        "adv_conjunction": {"items_per_condition": 600, "conditions": ["S", "P"]},
        "namepp": {"items_per_condition": 900, "conditions": ["S", "P"]},
        "nounpp": {"items_per_condition": 600, "conditions": ["SS", "SP", "PS", "PP"]},
        "nounpp_adv": {
            "items_per_condition": 600,
            "conditions": ["SS", "SP", "PS", "PP"],
        },
        "simple": {"items_per_condition": 300, "conditions": ["S", "P"]},
    }

    def __init__(
        self,
        model: LanguageModel,
        vocab_path: str,
        corpus_path: str,
        subtasks: Optional[List[str]] = None,
    ):
        super().__init__(model, vocab_path, corpus_path, subtasks=subtasks)

    def initialize(
        self, corpus_path: str, subtasks: Optional[List[str]] = None
    ) -> DownstreamCorpora:
        """ Performs the initialization for the tasks of
        Marvin & Linzen (2018)

        Arxiv link: https://arxiv.org/pdf/1808.09031.pdf

        Repo: https://github.com/BeckyMarvin/LM_syneval

        Parameters
        ----------
        corpus_path : str
            Path to directory containing the Marvin datasets that can be
            found in the github repo.
        subtasks : List[str], optional
            The downstream tasks that will be tested. If not provided this
            will default to the full set of conditions.

        Returns
        -------
        corpora : Dict[str, Corpus]
            Dictionary mapping a subtask to a Corpus.
        """
        subtasks = subtasks or self.descriptions.keys()

        corpora: DownstreamCorpora = {}

        for subtask in subtasks:
            items_per_condition = self.descriptions[subtask]["items_per_condition"]

            for i, condition in enumerate(self.descriptions[subtask]["conditions"]):
                corpus = Corpus.create(
                    os.path.join(corpus_path, f"{subtask}.txt"),
                    header=["sen", "type", "correct", "idx"],
                    vocab_path=self.vocab_path,
                )

                condition_slice = slice(
                    i * items_per_condition, (i + 1) * items_per_condition
                )
                self.postprocess_corpus(corpus, condition_slice)

                corpora.setdefault(subtask, {})[condition] = corpus

        return corpora

    @staticmethod
    def postprocess_corpus(corpus: Corpus, condition_slice: slice) -> None:
        """ Set the correct and incorrect verb for each sentence. """
        corpus.fields["token"] = RawField()
        corpus.fields["wrong_token"] = RawField()

        corpus.fields["token"].is_target = False
        corpus.fields["wrong_token"].is_target = False

        for idx in range(0, len(corpus), 2):
            setattr(corpus[idx], "token", [corpus[idx].sen[-1]])
            setattr(corpus[idx], "wrong_token", [corpus[idx + 1].sen[-1]])
            corpus[idx].sen = corpus[idx].sen[:-1]

        corpus.examples = corpus.examples[::2]

        # Slice up the corpus into the partition of the current condition
        corpus.examples = corpus.examples[condition_slice]
