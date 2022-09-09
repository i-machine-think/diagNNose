import glob
import os
from typing import Dict, List, Optional

import pandas as pd
from torchtext.data import Example

from diagnnose.corpus import Corpus
from diagnnose.typedefs.syntax import SyntaxEvalCorpora

from ..task import SyntaxEvalTask


class BlimpTask(SyntaxEvalTask):
    def initialize(
        self,
        path: str,
        subtasks: Optional[List[str]] = None,
        compare_full_sen: bool = False,
    ) -> SyntaxEvalCorpora:
        """Performs the initialization for the BLiMP tasks of
        Warstadt et al. (2020)

        Arxiv link: https://arxiv.org/pdf/1912.00582.pdf

        Repo: https://github.com/alexwarstadt/blimp

        Parameters
        ----------
        path : str
            Path to directory containing the BLiMP datasets that can be
            found in the github repo.
        subtasks : List[str], optional
            The downstream tasks that will be tested. If not provided
            this will default to the full set of conditions.
        compare_full_sen : bool, optional
            Toggle to compare minimal pairs based on the full sentence
            probabilities. Otherwise the one- or two-prefix method will
            be used instead, where applicable.

        Returns
        -------
        corpora : Dict[str, Corpus]
            Dictionary mapping a subtask to a Corpus.
        """
        subtask_paths = glob.glob(os.path.join(path, "*.jsonl"))
        all_subtasks = [os.path.basename(path).split(".")[0] for path in subtask_paths]
        subtask_to_path = dict(zip(all_subtasks, subtask_paths))

        subtasks: List[str] = subtasks or all_subtasks

        corpora: SyntaxEvalCorpora = {}

        for subtask in subtasks:
            subtask_path = subtask_to_path[subtask]
            subtask_corpus = self._initialize_subtask(subtask_path, compare_full_sen)

            if subtask_corpus is not None:
                corpora[subtask] = subtask_corpus

        return corpora

    def _initialize_subtask(
        self, subtask_path: str, compare_full_sen: bool
    ) -> Optional[Corpus]:
        raw_corpus = pd.read_json(path_or_buf=subtask_path, lines=True)

        header = raw_corpus.keys().tolist()

        if compare_full_sen:
            header[header.index("sentence_good")] = "sen"
            header[header.index("sentence_bad")] = "counter_sen"
            tokenize_columns = ["sen", "counter_sen"]
        elif "one_prefix_prefix" in raw_corpus:
            header[header.index("one_prefix_prefix")] = "sen"
            header[header.index("one_prefix_word_good")] = "token"
            header[header.index("one_prefix_word_bad")] = "counter_token"
            tokenize_columns = ["sen"]
        elif "two_prefix_prefix_good" in raw_corpus:
            header[header.index("two_prefix_prefix_good")] = "sen"
            header[header.index("two_prefix_prefix_bad")] = "counter_sen"
            header[header.index("two_prefix_word")] = "token"
            tokenize_columns = ["sen", "counter_sen"]
        else:
            return None

        fields = Corpus.create_fields(
            header, tokenize_columns=tokenize_columns, tokenizer=self.tokenizer
        )

        examples = [
            Example.fromlist(item, fields) for item in raw_corpus.values.tolist()
        ]

        corpus = Corpus(examples, fields)

        return corpus
