import random
from typing import Dict, Optional, Tuple

import torch

from diagnnose.activations.activation_reader import ActivationReader
from diagnnose.activations.selection_funcs import return_all, union
from diagnnose.corpus import Corpus
from diagnnose.corpus.create_labels import create_labels_from_corpus
from diagnnose.extract import simple_extract
from diagnnose.models import LanguageModel
from diagnnose.typedefs.activations import (
    ActivationIndex,
    ActivationName,
    ActivationNames,
    SelectionFunc,
)
from diagnnose.typedefs.probe import ControlTask, DataDict, DataSplit


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
    corpus : Corpus
        Corpus containing the labels for each sentence.
    activations_dir : str, optional
        Directory containing the extracted activations. If not provided,
        new activations will be extracted. If ``create_new_activations``
        is set to ``True``, the newly extracted activations will be
        stored in this directory.
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
    create_new_activations : bool, optional
        Toggle to create new activations based on corpus and model.
        Overwrites existing activations that might be present in
        ``activations_dir``.
    """

    def __init__(
        self,
        corpus: Corpus,
        activations_dir: Optional[str] = None,
        model: Optional[LanguageModel] = None,
        activation_names: Optional[ActivationNames] = None,
        train_selection_func: Optional[SelectionFunc] = None,
        test_activations_dir: Optional[str] = None,
        test_corpus: Optional[Corpus] = None,
        test_selection_func: Optional[SelectionFunc] = None,
        control_task: Optional[ControlTask] = None,
        train_test_ratio: Optional[float] = None,
        create_new_activations: bool = False,
    ) -> None:
        assert (
            None not in [test_corpus, test_selection_func]
            or test_selection_func is not None
            or train_test_ratio is not None
        ), (
            "Test split must be provided by either passing a test_corpus, a test_selection_func, "
            "or a train_test_ratio. The docstring of DataLoader contains more precise information."
        )

        self.activation_names = activation_names
        self.create_new_activations = create_new_activations

        selection_func = (
            union((train_selection_func, test_selection_func))
            if test_selection_func is not None and test_corpus is None
            else train_selection_func
        )

        train_activation_reader = self._create_activation_reader(
            model, corpus, activations_dir, selection_func
        )

        self.train_ids, self.test_ids = self._train_test_ids(
            corpus,
            train_activation_reader.selection_func,
            train_selection_func,
            test_selection_func if test_corpus is None else None,
            train_test_ratio if test_corpus is None else None,
        )

        self.train_split = self._create_data_split(
            train_activation_reader,
            corpus,
            self.train_ids,
            control_task,
        )

        if test_corpus is not None:
            test_activation_reader = self._create_test_split(
                model,
                test_activations_dir,
                test_corpus,
                test_selection_func,
            )
        else:
            test_activation_reader = train_activation_reader
            test_corpus = corpus

        self.test_split = self._create_data_split(
            test_activation_reader,
            test_corpus,
            self.test_ids,
            control_task,
        )

        self.label_vocab: Dict[str, int] = corpus.fields[corpus.labels_column].vocab

        assert len(self.test_split.labels) > 0, (
            "DataSplit contains no test labels, check whether test_selection_func, "
            "test_activation_dir or train_test_ratio are set up correctly"
        )

    def _create_activation_reader(
        self,
        model: Optional[LanguageModel],
        corpus: Corpus,
        activations_dir: str,
        selection_func: Optional[SelectionFunc],
    ) -> ActivationReader:
        if activations_dir is None or self.create_new_activations:
            assert model is not None

            activation_reader, _ = simple_extract(
                model,
                corpus,
                self.activation_names,
                activations_dir=activations_dir,
                selection_func=selection_func or return_all,
            )
            activation_reader.cat_activations = True
        else:
            activation_reader = ActivationReader(activations_dir, cat_activations=True)

        return activation_reader

    @staticmethod
    def _create_data_split(
        activation_reader: ActivationReader,
        corpus: Corpus,
        label_ids: ActivationIndex,
        control_task: Optional[ControlTask],
    ) -> DataSplit:
        """Creates a labels Tensor and returns a DataSplit.

        The DataSplit contains an ActivationReader, the labels, and,
        optionally, the control labels.
        """
        labels = create_labels_from_corpus(
            corpus,
            selection_func=activation_reader.selection_func,
        )
        labels = labels[label_ids]

        if control_task is not None:
            control_labels = create_labels_from_corpus(
                corpus,
                selection_func=activation_reader.selection_func,
                control_task=control_task,
            )
            control_labels = control_labels[label_ids]
        else:
            control_labels = None

        return DataSplit(activation_reader, labels, control_labels)

    def _create_test_split(
        self,
        model: Optional[LanguageModel],
        test_activations_dir: Optional[str],
        test_corpus: Optional[Corpus],
        test_selection_func: Optional[SelectionFunc],
    ) -> ActivationReader:
        """Creates the DataSplit for the test items that are retrieved
        from a different Corpus than the train items.
        """
        test_activation_reader = self._create_activation_reader(
            model,
            test_corpus,
            test_activations_dir,
            test_selection_func,
        )

        self.test_ids, _ = self._train_test_ids(
            test_corpus,
            test_activation_reader.selection_func,
            test_selection_func,
            None,
            None,
        )

        return test_activation_reader

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

        data_dict = DataDict(
            train_activations,
            train_labels,
            train_control_labels,
            test_activations,
            test_labels,
            test_control_labels,
        )

        return data_dict

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
