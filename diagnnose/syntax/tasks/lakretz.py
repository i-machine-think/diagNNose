import os
from typing import Dict, List, Optional

from diagnnose.corpus import Corpus
from diagnnose.typedefs.syntax import SyntaxEvalCorpora

from ..task import SyntaxEvalTask


class LakretzTask(SyntaxEvalTask):
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

    def initialize(
        self, path: str, subtasks: Optional[List[str]] = None
    ) -> SyntaxEvalCorpora:
        """Performs the initialization for the tasks of
        Marvin & Linzen (2018)

        Arxiv link: https://arxiv.org/pdf/1808.09031.pdf

        Repo: https://github.com/BeckyMarvin/LM_syneval

        Parameters
        ----------
        path : str
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

        corpora: SyntaxEvalCorpora = {}

        for subtask in subtasks:
            items_per_condition = self.descriptions[subtask]["items_per_condition"]

            for i, condition in enumerate(self.descriptions[subtask]["conditions"]):
                start_idx = i * items_per_condition
                stop_idx = (i + 1) * items_per_condition
                condition_slice = slice(start_idx, stop_idx)

                corpus = self._create_corpus(
                    os.path.join(path, f"{subtask}.txt"), condition_slice
                )

                corpora.setdefault(subtask, {})[condition] = corpus

        return corpora

    def _create_corpus(self, path: str, condition_slice: slice) -> Corpus:
        """Attach the correct and incorrect verb form to each sentence
        in the corpus.
        """
        raw_corpus = Corpus.create_raw_corpus(path)

        for idx in range(0, len(raw_corpus), 2):
            token = raw_corpus[idx][0].split()[-1]
            counter_token = raw_corpus[idx + 1][0].split()[-1]
            sen = " ".join(raw_corpus[idx][0].split()[:-1])
            raw_corpus[idx] = [sen, token, counter_token]

        raw_corpus = raw_corpus[::2][condition_slice]

        fields = Corpus.create_fields(
            ["sen", "token", "counter_token"], tokenizer=self.tokenizer
        )

        examples = Corpus.create_examples(raw_corpus, fields)

        return Corpus(examples, fields)
