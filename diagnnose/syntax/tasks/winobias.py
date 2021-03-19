import os
from typing import Dict, List, Optional

from torchtext.data import RawField

from diagnnose.corpus import Corpus
from diagnnose.typedefs.syntax import SyntaxEvalCorpora

from ..task import SyntaxEvalTask


class WinobiasTask(SyntaxEvalTask):
    def initialize(
        self, path: str, subtasks: Optional[List[str]] = None
    ) -> SyntaxEvalCorpora:
        """

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
        subtasks = subtasks or ["stereo", "unamb"]

        corpora: SyntaxEvalCorpora = {}

        for subtask in subtasks:
            for condition in ["FF", "FM", "MF", "MM"]:
                corpus = Corpus.create(
                    os.path.join(path, f"{subtask}_{condition}.tsv"),
                    header_from_first_line=True,
                    tokenizer=self.tokenizer,
                )

                self._add_output_classes(corpus)

                corpora.setdefault(subtask, {})[condition] = corpus

        return corpora

    @staticmethod
    def _add_output_classes(corpus: Corpus) -> None:
        """ Set the the pronouns for each sentence. """
        corpus.fields["token"] = RawField()
        corpus.fields["counter_token"] = RawField()

        corpus.fields["token"].is_target = False
        corpus.fields["counter_token"].is_target = False

        for ex in corpus:
            setattr(ex, "token", "he")
            setattr(ex, "counter_token", "she")
