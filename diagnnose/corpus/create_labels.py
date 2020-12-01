from typing import Optional

import torch
from torch import Tensor

from diagnnose.corpus import Corpus
from diagnnose.typedefs.activations import SelectionFunc
from diagnnose.typedefs.probe import ControlTask


def create_labels_from_corpus(
    corpus: Corpus,
    selection_func: SelectionFunc = lambda sen_id, pos, example: True,
    control_task: Optional[ControlTask] = None,
) -> Tensor:
    """Creates labels based on the selection_func that was used during
    extraction.

    Parameters
    ----------
    corpus : Corpus
        Labeled corpus containing sentence and label information.
    selection_func: SelectFunc, optional
        Function that determines whether a label should be stored.
    control_task: ControlTask, optional
        Control task function of Hewitt et al. (2019), mapping a corpus
        item to a random label.
    """
    all_labels = []

    for item in corpus.examples:
        label_idx = 0

        sen = getattr(item, corpus.sen_column)
        labels = getattr(item, corpus.labels_column)
        if isinstance(labels, str):
            labels = labels.split()

        each_token_labeled = len(sen) == len(labels)

        for wpos in range(len(sen)):
            if selection_func(wpos, item):
                if control_task is not None:
                    label = control_task(wpos, item)
                else:
                    label = labels[label_idx]

                all_labels.append(label)

                if not each_token_labeled and len(labels) > 1:
                    label_idx += 1
            if each_token_labeled:
                label_idx += 1

    # Create new label vocab that only contains the labels that have been selected
    label_vocab = {label: idx for idx, label in enumerate(set(all_labels))}
    corpus.fields[corpus.labels_column].vocab = label_vocab

    return torch.tensor([label_vocab[label] for label in all_labels])
