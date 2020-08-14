import os
from typing import Dict, List, Optional

from torchtext.data import RawField

from diagnnose.corpus import Corpus
from diagnnose.models import LanguageModel
from diagnnose.tokenizer import Tokenizer

from .task import DownstreamCorpora, DownstreamTask


class WinobiasDownstream(DownstreamTask):
    def __init__(
        self,
        model: LanguageModel,
        tokenizer: Tokenizer,
        corpus_path: str,
        subtasks: Optional[List[str]] = None,
    ):
        super().__init__(model, tokenizer, corpus_path, subtasks=subtasks)

    def initialize(
        self, corpus_path: str, subtasks: Optional[List[str]] = None
    ) -> DownstreamCorpora:
        """

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
        subtasks = subtasks or ["stereo", "unamb"]

        corpora: DownstreamCorpora = {}

        for subtask in subtasks:
            for condition in ["FF", "FM", "MF", "MM"]:
                corpus = Corpus.create(
                    os.path.join(corpus_path, f"{subtask}_{condition}.tsv"),
                    header_from_first_line=True,
                    tokenizer=self.tokenizer,
                )

                self.add_output_classes(corpus)

                corpora.setdefault(subtask, {})[condition] = corpus

        return corpora

    @staticmethod
    def add_output_classes(corpus: Corpus) -> None:
        """ Set the correct and incorrect verb for each sentence. """
        corpus.fields["token"] = RawField()
        corpus.fields["wrong_token"] = RawField()

        corpus.fields["token"].is_target = False
        corpus.fields["wrong_token"].is_target = False

        for ex in corpus:
            setattr(ex, "token", ["he"])
            setattr(ex, "wrong_token", ["she"])
