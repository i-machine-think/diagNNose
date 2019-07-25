import numpy as np
from diagnnose.typedefs.activations import SelectFunc
from diagnnose.typedefs.corpus import Corpus


def create_labels_from_corpus(
    corpus: Corpus,
    selection_func: SelectFunc = lambda sen_id, pos, example: True,
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
    for i, item in enumerate(corpus.examples):
        for j in range(len(item.sen)):
            if selection_func(i, j, item):
                labels.append(item.labels[j])

    return np.array(labels)
