from typing import Dict, List, Optional

from torchtext.data import Example

from diagnnose.corpus import Corpus
from diagnnose.typedefs.models import LanguageModel
from diagnnose.vocab import create_vocab, W2I

from .warstadt_preproc import create_downstream_corpus, preproc_warstadt, ENVS
from ..task import DownstreamTask, DownstreamCorpora


class WarstadtDownstream(DownstreamTask):
    """

    Parameters
    ----------
    use_full_model_probs : bool, optional
        Toggle to calculate the full model probs for the NPI sentences.
        If set to False only the NPI logits will be compared, instead
        of their Softmax probabilities. Defaults to True.
    """

    def __init__(
        self,
        model: LanguageModel,
        vocab_path: str,
        corpus_path: str,
        subtasks: Optional[List[str]] = None,
        use_full_model_probs: bool = True,
    ):
        self.use_full_model_probs = use_full_model_probs

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
        subtasks: List[str] = subtasks or ENVS

        corpora: DownstreamCorpora = {}

        orig_corpus = preproc_warstadt(corpus_path)

        vocab = create_vocab(self.vocab_path)

        for env in subtasks:
            raw_corpus = create_downstream_corpus(orig_corpus, envs=[env])

            header = raw_corpus[0].split("\t")
            tokenize_columns = ["sen", "counter_sen"]
            fields = Corpus.create_fields(header, tokenize_columns=tokenize_columns)
            examples = [
                Example.fromlist(line.split("\t"), fields) for line in raw_corpus[1:]
            ]
            corpus = Corpus(examples, fields, tokenize_columns=tokenize_columns)
            corpus.attach_vocab(vocab, tokenize_columns)

            corpora[env] = corpus

        return corpora

    @staticmethod
    def calc_counter_sen(subtask: str) -> bool:
        return True
