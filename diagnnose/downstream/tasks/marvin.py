import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

from torchtext.data import Example, Field
from transformers import PreTrainedTokenizer

from diagnnose.corpus import Corpus
from diagnnose.models import LanguageModel
from diagnnose.utils.pickle import load_pickle

from .task import DownstreamCorpora, DownstreamTask


class MarvinDownstream(DownstreamTask):
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
        tokenizer: PreTrainedTokenizer,
        corpus_path: str,
        subtasks: Optional[List[str]] = None,
        use_full_model_probs: bool = True,
    ):
        self.use_full_model_probs = use_full_model_probs

        super().__init__(model, tokenizer, corpus_path, subtasks=subtasks)

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
        subtask_paths = glob.glob(os.path.join(corpus_path, "*.pickle"))
        all_subtasks = [os.path.basename(path).split(".")[0] for path in subtask_paths]
        subtask_to_path = dict(zip(all_subtasks, subtask_paths))

        subtasks: List[str] = subtasks or all_subtasks

        corpora: DownstreamCorpora = {}

        for subtask in subtasks:
            subtask_path = subtask_to_path[subtask]
            subtask_corpora: Dict[str, Corpus] = self.initialize_subtask(
                subtask, subtask_path
            )

            corpora[subtask] = subtask_corpora

        return corpora

    @staticmethod
    def calc_counter_sen(subtask: str) -> bool:
        return "npi" in subtask

    def initialize_subtask(self, subtask: str, corpus_path: str) -> Dict[str, Corpus]:
        corpus_dict: Dict[str, List[Sequence[str]]] = load_pickle(corpus_path)

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
            examples = self.create_examples(subtask, sens, fields)

            corpus = Corpus(examples, fields)

            subtask_corpora[condition] = corpus

        return subtask_corpora

    def create_examples(
        self, subtask: str, sens: List[Sequence[str]], fields: List[Tuple[str, Field]]
    ):
        if "npi" in subtask:
            return self.create_npi_examples(sens, fields)

        return self.create_sva_examples(sens, fields)

    @staticmethod
    def create_sva_examples(
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
    def create_npi_examples(
        sens: List[Sequence[str]], fields: List[Tuple[str, Field]]
    ) -> List[Example]:
        examples = []
        subsens_seen = set()

        for s1, s2, _ in sens:
            s1, s2 = s1.split(), s2.split()

            npi = "ever"
            ever_idx = s1.index(npi)
            subsen = " ".join(s1[:ever_idx])

            if subsen in subsens_seen:
                continue

            ex = Example.fromlist([s1[:ever_idx], s2[:ever_idx], npi], fields)
            examples.append(ex)
            subsens_seen.add(subsen)

        return examples
