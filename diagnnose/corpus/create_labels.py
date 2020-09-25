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
    labels = []
    label_vocab = corpus.fields["labels"].tokenizer.stoi
    for idx, item in enumerate(corpus.examples):
        label_idx = 0
        each_token_labeled = len(item.sen) == len(item.labels)
        for wpos in range(len(item.sen)):
            if selection_func(idx, wpos, item):
                if control_task is not None:
                    label = control_task(idx, wpos, item)
                else:
                    try:
                        label = item.labels[label_idx]
                    except IndexError:
                        print(item)
                        raise
                if isinstance(label, str):
                    label = label_vocab[label]

                labels.append(label)

                if not each_token_labeled:
                    label_idx += 1
            if each_token_labeled:
                label_idx += 1

    return torch.tensor(labels)
