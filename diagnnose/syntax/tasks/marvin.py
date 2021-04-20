import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

from torchtext.data import Example, Field

from diagnnose.corpus import Corpus
from diagnnose.typedefs.syntax import SyntaxEvalCorpora
from diagnnose.utils.pickle import load_pickle

from ..task import SyntaxEvalTask


class MarvinTask(SyntaxEvalTask):
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
        subtask_paths = glob.glob(os.path.join(path, "*.pickle"))
        all_subtasks = [os.path.basename(path).split(".")[0] for path in subtask_paths]
        subtask_to_path = dict(zip(all_subtasks, subtask_paths))

        subtasks: List[str] = subtasks or all_subtasks

        corpora: SyntaxEvalCorpora = {}

        for subtask in subtasks:
            subtask_path = subtask_to_path[subtask]
            subtask_corpora: Dict[str, Corpus] = self._initialize_subtask(
                subtask, subtask_path
            )

            corpora[subtask] = subtask_corpora

        return corpora

    def _initialize_subtask(self, subtask: str, subtask_path: str) -> Dict[str, Corpus]:
        corpus_dict: Dict[str, List[Sequence[str]]] = load_pickle(subtask_path)

        if "npi" in subtask:
            header = ["sen", "counter_sen", "token"]
            tokenize_columns = ["sen", "counter_sen"]
        else:
            header = ["sen", "token", "counter_token"]
            tokenize_columns = ["sen"]

        fields = Corpus.create_fields(
            header, tokenize_columns=tokenize_columns, tokenizer=self.tokenizer
        )
        subtask_corpora: Dict[str, Corpus] = {}

        for condition, sens in corpus_dict.items():
            examples = self._create_examples(subtask, sens, fields)

            corpus = Corpus(examples, fields)

            subtask_corpora[condition] = corpus

        return subtask_corpora

    def _create_examples(
        self, subtask: str, sens: List[Sequence[str]], fields: List[Tuple[str, Field]]
    ):
        if "npi" in subtask:
            return self._create_npi_examples(sens, fields)

        return self._create_sva_examples(sens, fields)

    @staticmethod
    def _create_sva_examples(
        sens: List[Sequence[str]], fields: List[Tuple[str, Field]]
    ) -> List[Example]:
        examples = []

        for s1, s2 in sens:
            s1, s2 = s1.split(), s2.split()

            # Locate index of verb as first point where correct and incorrect sentence differ.
            verb_index = 0
            for w1, w2 in zip(s1, s2):
                if w1 != w2:
                    break
                verb_index += 1

            subsen = s1[:verb_index]
            verb = s1[verb_index]
            wrong_verb = s2[verb_index]
            ex = Example.fromlist([subsen, verb, wrong_verb], fields)
            examples.append(ex)

        return examples

    @staticmethod
    def _create_npi_examples(
        sens: List[Sequence[str]], fields: List[Tuple[str, Field]]
    ) -> List[Example]:
        examples = []

        for s1, s2, _ in sens:
            s1, s2 = s1.split(), s2.split()

            npi = "ever"
            ever_idx = s1.index(npi)
            sen = " ".join(s1[:ever_idx])
            counter_sen = " ".join(s2[:ever_idx])

            ex = Example.fromlist([sen, counter_sen, npi], fields)
            examples.append(ex)

        return examples
