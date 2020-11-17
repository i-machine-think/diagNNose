from typing import Dict, List, Optional

from torchtext.data import Example

from diagnnose.corpus import Corpus
from diagnnose.typedefs.syntax import SyntaxEvalCorpora

from ..task import SyntaxEvalTask
from .warstadt_preproc import ENVS, create_downstream_corpus, preproc_warstadt


class WarstadtTask(SyntaxEvalTask):
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
        subtasks: List[str] = subtasks or ENVS

        corpora: SyntaxEvalCorpora = {}

        orig_corpus = preproc_warstadt(path)

        for env in subtasks:
            raw_corpus = create_downstream_corpus(orig_corpus, envs=[env])

            header = raw_corpus[0].split("\t")
            tokenize_columns = ["sen", "counter_sen"]
            fields = Corpus.create_fields(
                header, tokenize_columns=tokenize_columns, tokenizer=self.tokenizer
            )
            examples = [
                Example.fromlist(line.split("\t"), fields) for line in raw_corpus[1:]
            ]
            corpus = Corpus(examples, fields)

            corpora[env] = corpus

        return corpora
