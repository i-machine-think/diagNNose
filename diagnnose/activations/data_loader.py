from typing import Optional

import numpy as np

from diagnnose.activations.activation_reader import ActivationReader
from diagnnose.corpus.create_labels import create_labels_from_corpus
from diagnnose.typedefs.activations import ActivationName, SelectFunc
from diagnnose.typedefs.classifiers import DataDict
from diagnnose.typedefs.corpus import Corpus, Labels


class DataLoader:
    """ Reads in pickled activations that have been extracted.

    Parameters
    ----------
    activations_dir : str
        Directory containing the extracted activations
    corpus : Corpus
        Corpus containing the labels for each sentence.

    Attributes
    ----------
    activations_dir : str
    labels : Optional[np.ndarray]
        Numpy array containing the extracted labels. Accessed by the
        property self.labels.
    """

    def __init__(
        self,
        activations_dir: str,
        corpus: Optional[Corpus] = None,
        labels: Optional[Labels] = None,
        selection_func: SelectFunc = lambda sen_id, pos, example: True
    ) -> None:

        assert (
            corpus is not None or labels is not None
        ), "Either `corpus` or `labels` should be provided!"

        self.activation_reader = ActivationReader(activations_dir)
        self.data_len = len(self.activation_reader)

        if labels is None:
            self.labels = create_labels_from_corpus(corpus, selection_func=selection_func)
        else:
            self.labels = labels

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
            Percentage of the train/test split. Defaults to 0.9.
        """

        if data_subset_size != -1:
            assert (
                0 < data_subset_size <= self.data_len
            ), "Size of subset must not be bigger than the full data set."

        activations = self.activation_reader.read_activations(activation_name)

        data_size = self.data_len if data_subset_size == -1 else data_subset_size
        split = int(data_size * train_test_split)

        indices = np.random.choice(range(data_size), data_size, replace=False)
        train_indices = indices[:split]
        test_indices = indices[split:]

        return {
            "train_x": activations[train_indices],
            "train_y": self.labels[train_indices],
            "test_x": activations[test_indices],
            "test_y": self.labels[test_indices],
        }
