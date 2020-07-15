from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torchtext.vocab import Vocab

from diagnnose.activations.activation_reader import ActivationReader
from diagnnose.corpus.create_labels import create_labels_from_corpus
from diagnnose.typedefs.activations import ActivationName, SelectionFunc
from diagnnose.typedefs.classifiers import ControlTask, DataDict
from diagnnose.corpus import Corpus


class DataLoader:
    """ Reads in pickled activations that have been extracted.

    Parameters
    ----------
    activations_dir : str
        Directory containing the extracted activations.
    corpus : Corpus
        Corpus containing the labels for each sentence.
    test_activations_dir : str, optional
        Directory containing the extracted test activations. If not
        provided the train activation set will be split and partially
        used as test set.
    test_corpus : Corpus, optional
        Corpus containing the test labels for each sentence. Must be
        provided if `test_activations_dir` is provided.
    train_selection_func : SelectFunc, optional
        Selection function that determines whether a corpus item should
        be taken into account for training. If not provided all
        extracted activations will be used and split into a random
        train/test split.
    test_selection_func : SelectFunc, optional
        Selection function that determines whether a corpus item should
        be taken into account for testing. If not provided all
        extracted activations will be used and split into a random
        train/test split.
    control_task: ControlTask, optional
        Control task function of Hewitt et al. (2019), mapping a corpus
        item to a random label.
    """

    # TODO: Move init logic to own method
    def __init__(
        self,
        activations_dir: str,
        corpus: Corpus,
        test_activations_dir: Optional[str] = None,
        test_corpus: Optional[Corpus] = None,
        train_selection_func: SelectionFunc = lambda sen_id, pos, example: True,
        test_selection_func: Optional[SelectionFunc] = None,
        control_task: Optional[ControlTask] = None,
    ) -> None:
        assert corpus is not None, "`corpus`should be provided!"

        self.train_labels = create_labels_from_corpus(
            corpus, selection_func=train_selection_func
        )
        self.test_labels = None
        self.train_labels_control = None
        self.test_labels_control = None
        if control_task is not None:
            self.train_labels_control = create_labels_from_corpus(
                corpus, selection_func=train_selection_func, control_task=control_task
            )

        self.label_vocab: Vocab = corpus.fields["labels"].vocab

        if test_activations_dir is not None:
            self.test_activation_reader = ActivationReader(test_activations_dir)
            test_selection_func = test_selection_func or (
                lambda sen_id, pos, example: True
            )
            assert test_corpus is not None, "`test_corpus` should be provided!"
            self.test_labels = create_labels_from_corpus(
                test_corpus, selection_func=test_selection_func
            )
            if control_task is not None:
                self.test_labels_control = create_labels_from_corpus(
                    test_corpus,
                    selection_func=test_selection_func,
                    control_task=control_task,
                )
        else:
            self.test_activation_reader = None

        if test_selection_func is not None:
            self.test_labels = create_labels_from_corpus(
                corpus, selection_func=test_selection_func
            )
            if control_task is not None:
                self.test_labels_control = create_labels_from_corpus(
                    corpus,
                    selection_func=test_selection_func,
                    control_task=control_task,
                )

        self.activation_reader = ActivationReader(activations_dir)

        orig_selection_func = self.activation_reader.selection_func

        self.train_ids, self.test_ids = self._create_train_test_mask(
            corpus, orig_selection_func, train_selection_func, test_selection_func
        )

    def create_data_split(
        self,
        activation_name: ActivationName,
        data_subset_size: int = -1,
        train_test_split: float = 0.9,
    ) -> DataDict:
        """ Creates train/test data split of activations

        Parameters
        ----------
        activation_name : ActivationName
            (layer, name) tuple indicating the activations to be read in
        data_subset_size : int, optional
            Subset size of data to train on. Defaults to -1, indicating
            the entire data set.
        train_test_split : float
            Percentage of the train/test split. If separate test
            activations are provided this split won't be used.
            Defaults to 0.9/0.1.

        Returns
        -------
        data_dict : DataDict
            Dictionary mapping train and test embeddings (train_x,
            test_x) to corpus labels (train_y, test_y), and optionally
            to control task labels (train_y_control, test_y_control).
        """

        train_activations = self.activation_reader.read_activations(activation_name)

        test_activations = train_activations[self.test_ids]
        train_activations = train_activations[self.train_ids]

        if data_subset_size == -1:
            data_size = train_activations.size(0)
        else:
            data_size = min(data_subset_size, train_activations.size(0))

        # Shuffle activations
        indices = np.random.choice(
            range(train_activations.size(0)), data_size, replace=False
        )

        train_activations = train_activations[indices]
        train_labels = self.train_labels[indices]
        test_labels = self.test_labels

        train_labels_control = self.train_labels_control
        test_labels_control = self.test_labels_control
        if train_labels_control is not None:
            train_labels_control = train_labels_control[indices]

        if self.test_activation_reader is not None:
            test_activations = self.test_activation_reader.read_activations(
                activation_name
            )
            test_labels = self.test_labels
        # Create test set from split in training data
        elif test_activations.size(0) == 0:
            split = int(data_size * train_test_split)

            test_activations = train_activations[split:]
            test_labels = train_labels[split:]
            train_activations = train_activations[:split]
            train_labels = train_labels[:split]

            if train_labels_control is not None:
                test_labels_control = train_labels_control[split:]
                train_labels_control = train_labels_control[:split]

        return {
            "train_x": train_activations,
            "train_y": train_labels,
            "train_y_control": train_labels_control,
            "test_x": test_activations,
            "test_y": test_labels,
            "test_y_control": test_labels_control,
        }

    @staticmethod
    def _create_train_test_mask(
        corpus: Corpus,
        orig_selection_func: SelectionFunc,
        train_selection_func: SelectionFunc,
        test_selection_func: Optional[SelectionFunc],
    ) -> Tuple[Tensor, Tensor]:
        """ Creates a tensor mask for the train/test split.

        Mask is created based on the provided selection functions.
        Note that the train_selection_func takes precedence over the
        test_selection_func. The mask also takes the original
        selection_func into account, and will skip items that are
        not part of neither the provided selection_funcs.
        """
        train_ids = []
        test_ids = []

        for idx, item in enumerate(corpus.examples):
            for pos in range(len(item.sen)):
                if orig_selection_func(idx, pos, item):
                    if train_selection_func(idx, pos, item):
                        train_ids.append(True)
                        test_ids.append(False)
                    elif test_selection_func and test_selection_func(idx, pos, item):
                        train_ids.append(False)
                        test_ids.append(True)
                    else:
                        train_ids.append(False)
                        test_ids.append(False)

        return (
            torch.tensor(train_ids).to(torch.uint8),
            torch.tensor(test_ids).to(torch.uint8),
        )
