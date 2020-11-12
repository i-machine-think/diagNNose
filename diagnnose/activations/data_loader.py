import random
from collections import namedtuple
from typing import Optional, Tuple

import torch
from torchtext.vocab import Vocab

from diagnnose.corpus import Corpus
from diagnnose.corpus.create_labels import create_labels_from_corpus
from diagnnose.typedefs.activations import (
    ActivationIndex,
    ActivationName,
    SelectionFunc,
)
from diagnnose.typedefs.probe import ControlTask, DataDict

from .activation_reader import ActivationReader


DataSplit = namedtuple("DataSplit", ["activation_reader", "labels", "control_labels"])


class DataLoader:
    """Reads in pickled activations that have been extracted, and
    creates a train/test split of activations and labels.

    Train/test split can be created in multiple ways:
    1. Using the test activations from a different corpus
    2. Using the activations from the same corpus for both, but the
    train/test split is defined on 2 separate selection_funcs.
    3. Based on a random 90/10 split.

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
    train_test_ratio : float, optional
        Ratio of the train/test split. If separate test
        activations are provided this split won't be used.
        Defaults to None, but must be provided if no test_selection_func
        is passed.
    """

    def __init__(
        self,
        activations_dir: str,
        corpus: Corpus,
        labels_column: str,
        test_activations_dir: Optional[str] = None,
        test_corpus: Optional[Corpus] = None,
        test_labels_column: Optional[str] = None,
        train_selection_func: Optional[SelectionFunc] = None,
        test_selection_func: Optional[SelectionFunc] = None,
        control_task: Optional[ControlTask] = None,
        train_test_ratio: Optional[float] = None,
    ) -> None:
        assert (
            (test_corpus is not None and test_activations_dir is not None)
            or test_selection_func is not None
            or train_test_ratio is not None
        )

        train_activation_reader = ActivationReader(
            activations_dir, cat_activations=True
        )

        self.train_ids, self.test_ids = self._train_test_ids(
            corpus,
            train_activation_reader.selection_func,
            train_selection_func,
            test_selection_func,
            train_test_ratio,
        )

        self.train_split = self._create_data_split(
            train_activation_reader,
            corpus,
            labels_column,
            self.train_ids,
            control_task,
        )

        self.test_split = self._create_test_split(
            corpus,
            labels_column,
            train_activation_reader,
            test_activations_dir,
            test_corpus,
            test_labels_column,
            test_selection_func,
            control_task,
            train_test_ratio,
        )

        self.label_vocab: Vocab = corpus.fields[labels_column].vocab

        assert len(self.test_split.labels) > 0, (
            "DataSplit contains no test labels, check whether test_selection_func, "
            "test_activation_dir or train_test_ratio are set up correctly"
        )

    @staticmethod
    def _create_data_split(
        activation_reader: ActivationReader,
        corpus: Corpus,
        labels_column: str,
        label_ids: ActivationIndex,
        control_task: Optional[ControlTask],
    ) -> DataSplit:
        """Creates a labels Tensor and returns a DataSplit.

        The DataSplit contains an ActivationReader, the labels, and,
        optionally, the control labels.
        """
        labels = create_labels_from_corpus(
            corpus,
            labels_column,
            selection_func=activation_reader.selection_func,
        )

        if control_task is not None:
            control_labels = create_labels_from_corpus(
                corpus,
                labels_column,
                selection_func=activation_reader.selection_func,
                control_task=control_task,
            )
        else:
            control_labels = None

        return DataSplit(
            activation_reader, labels[label_ids], control_labels[label_ids]
        )

    def _create_test_split(
        self,
        corpus: Corpus,
        labels_column: str,
        train_activation_reader: ActivationReader,
        test_activations_dir: Optional[str],
        test_corpus: Optional[Corpus],
        test_labels_column: Optional[str],
        test_selection_func: Optional[SelectionFunc],
        control_task: Optional[ControlTask],
        train_test_ratio: Optional[float],
    ) -> DataSplit:
        """Creates the DataSplit for the test items.

        Test items can be retrieved from a separate corpus, or from
        a subsplit of the training corpus. In the latter case the test
        items are either defined by a separate test_selection_func, or
        a random
        """
        if test_activations_dir is not None:
            test_activation_reader = ActivationReader(
                test_activations_dir, cat_activations=True
            )

            self.test_ids, _ = self._train_test_ids(
                test_corpus,
                test_activation_reader.selection_func,
                test_selection_func,
                None,
                train_test_ratio,
            )

            return self._create_data_split(
                test_activation_reader,
                test_corpus,
                (test_labels_column or labels_column),
                self.test_ids,
                control_task,
            )

        return self._create_data_split(
            train_activation_reader,
            corpus,
            labels_column,
            self.test_ids,
            control_task,
        )

    def load(
        self,
        activation_name: ActivationName,
    ) -> DataDict:
        """Creates train/test data split of activations

        Parameters
        ----------
        activation_name : ActivationName
            (layer, name) tuple indicating the activations to be read in

        Returns
        -------
        data_dict : DataDict
            Dictionary containing train and test activations, and their
            corresponding labels and, optionally, control labels.
        """
        train_activations = self.train_split.activation_reader[:, activation_name]
        test_activations = self.test_split.activation_reader[:, activation_name]

        train_activations = train_activations[self.train_ids]
        test_activations = test_activations[self.test_ids]

        train_labels = self.train_split.labels
        test_labels = self.test_split.labels

        train_control_labels = self.train_split.control_labels
        test_control_labels = self.test_split.control_labels

        return {
            "train_activations": train_activations,
            "train_labels": train_labels,
            "train_control_labels": train_control_labels,
            "test_activations": test_activations,
            "test_labels": test_labels,
            "test_control_labels": test_control_labels,
        }

    @staticmethod
    def _train_test_ids(
        corpus: Corpus,
        orig_selection_func: SelectionFunc,
        train_selection_func: Optional[SelectionFunc],
        test_selection_func: Optional[SelectionFunc],
        train_test_ratio: Optional[float],
    ) -> Tuple[ActivationIndex, ActivationIndex]:
        """Creates a tensor mask for the train/test split.

        Mask is created based on the provided selection functions.

        Note that the train_selection_func takes precedence over the
        test_selection_func.

        The mask also takes the original selection_func into account,
        and will skip items that are not part of either the provided
        selection_funcs.
        """
        train_ids = []
        test_ids = []

        for item in corpus.examples:
            sen = getattr(item, corpus.sen_column)
            for pos in range(len(sen)):
                if orig_selection_func(pos, item):
                    if train_selection_func is None or train_selection_func(pos, item):
                        in_train = (
                            random.random() < train_test_ratio
                            if train_test_ratio is not None
                            else True
                        )
                        train_ids.append(in_train)
                        test_ids.append(not in_train)
                    elif test_selection_func and test_selection_func(pos, item):
                        train_ids.append(False)
                        test_ids.append(True)
                    else:
                        train_ids.append(False)
                        test_ids.append(False)

        return (
            torch.tensor(train_ids).to(torch.bool),
            torch.tensor(test_ids).to(torch.bool),
        )
