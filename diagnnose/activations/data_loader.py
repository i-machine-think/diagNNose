from typing import Optional

import numpy as np

from diagnnose.activations.activation_reader import ActivationReader
from diagnnose.corpus.create_labels import create_labels_from_corpus
from diagnnose.typedefs.activations import ActivationName, SelectFunc
from diagnnose.typedefs.classifiers import DataDict
from diagnnose.typedefs.corpus import Corpus


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
    selection_func : SelectFunc, optional
        Selection function that determines whether a corpus item should
        be taken into account. If such a function has been used during
        extraction, make sure to pass it along here as well.
    test_selection_func : SelectFunc, optional
        Selection function that determines whether a corpus item should
        be taken into account for testing. If such a function has been
        used during extraction, make sure to pass it along here as well.
    """

    def __init__(
        self,
        activations_dir: str,
        corpus: Corpus,
        test_activations_dir: Optional[str] = None,
        test_corpus: Optional[Corpus] = None,
        selection_func: SelectFunc = lambda sen_id, pos, example: True,
        test_selection_func: SelectFunc = lambda sen_id, pos, example: True,
    ) -> None:
        assert corpus is not None, "`corpus`should be provided!"

        self.train_labels = create_labels_from_corpus(
            corpus, selection_func=selection_func
        )

        if test_activations_dir is not None:
            self.test_activation_reader = ActivationReader(test_activations_dir)
            assert test_corpus is not None, "`test_corpus` should be provided!"
            self.test_labels = create_labels_from_corpus(
                test_corpus, selection_func=test_selection_func
            )
        else:
            self.test_activation_reader = None
            self.test_labels = None

        self.activation_reader = ActivationReader(activations_dir)
        self.data_len = len(self.activation_reader)

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
        """

        if data_subset_size != -1:
            assert (
                0 < data_subset_size <= self.data_len
            ), "Size of subset can't be bigger than the full data set."

        train_activations = self.activation_reader.read_activations(activation_name)

        # Shuffle activations
        data_size = self.data_len if data_subset_size == -1 else data_subset_size
        indices = np.random.choice(range(data_size), data_size, replace=False)
        train_activations = train_activations[indices]
        train_labels = self.train_labels[indices]

        if self.test_activation_reader is not None:
            test_activations = self.test_activation_reader.read_activations(
                activation_name
            )
            test_labels = self.test_labels
        else:
            split = int(data_size * train_test_split)

            test_activations = train_activations[split:]
            test_labels = train_labels[split:]
            train_activations = train_activations[:split]
            train_labels = train_labels[:split]

        return {
            "train_x": train_activations,
            "train_y": train_labels,
            "test_x": test_activations,
            "test_y": test_labels,
        }
