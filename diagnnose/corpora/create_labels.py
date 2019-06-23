import numpy as np
from diagnnose.typedefs.corpus import Corpus
from diagnnose.typedefs.extraction import SelectFunc


def create_labels_from_corpus(corpus: Corpus,
                              selection_func: SelectFunc = lambda pos, token, labeled_sentence: True
                              ) -> np.ndarray:
    """ Creates labels based on the selection_func that was used during
    extraction.

    Parameters
    ----------
    corpus : Corpus
        Labeled corpus containing sentence and label information.
    selection_func: SelectFunc, optional
        Function that determines whether a label should be stored.
    """
    labels = []
    for s in corpus.values():
        for i, w in enumerate(s.sen):
            if selection_func(i, w, s):
                labels.append(s.labels[i])

    return np.array(labels)
