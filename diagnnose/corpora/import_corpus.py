from typing import Any, Dict
import os

from ..typedefs.corpus import LabeledCorpus, LabeledSentence
from ..utils.paths import load_pickle


def convert_to_labeled_corpus(corpus_path: str) -> LabeledCorpus:
    labeled_corpus = {}

    init_corpus: Dict[int, Dict[str, Any]] = load_pickle(os.path.expanduser(corpus_path))

    for key, item in init_corpus.items():
        assert 'sen' in item.keys() and 'labels' in item.keys(), 'Corpus item has wrong format.'

        sen = item['sen']
        labels = item['labels']
        misc_info = {k: v for k, v in item.items() if k not in ['sen', 'labels']}
        labeled_sentence = LabeledSentence(sen, labels, misc_info)
        labeled_sentence.validate()
        labeled_corpus[key] = labeled_sentence

    return labeled_corpus
