import pickle
from typing import Any, Dict

from ..typedefs.corpus import LabeledCorpus, LabeledSentence


def convert_to_labeled_corpus(corpus_path: str) -> LabeledCorpus:
    labeled_corpus = {}

    with open(corpus_path, 'rb') as f:
        corpus: Dict[int, Dict[str, Any]] = pickle.load(f)

    for key, item in corpus.items():
        assert 'sen' in item.keys() and 'labels' in item.keys(), 'Corpus item has wrong format.'

        sen = item['sen']
        labels = item['labels']
        misc_info = {k: v for k, v in item.items() if k not in ['sen', 'labels']}
        labeled_sentence = LabeledSentence(sen, labels, misc_info)
        labeled_sentence.validate()
        labeled_corpus[key] = labeled_sentence

    return labeled_corpus
