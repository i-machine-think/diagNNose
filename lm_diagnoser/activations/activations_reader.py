import pickle
from typing import Optional
from random import shuffle

import numpy as np

from ..typedefs.models import ActivationName
from ..typedefs.classifiers import DataDict
from ..utils.paths import load_pickle, trim


class ActivationsReader:
    def __init__(self,
                 activations_dir: str,
                 label_path: Optional[str]) -> None:

        self.activations_dir = trim(activations_dir)

        self.labels = self._read_labels(label_path)
        self.data_len = len(self.labels)

    def _read_labels(self, label_path: Optional[str]) -> np.array:
        if label_path is None:
            label_path = f'{self.activations_dir}/labels.pickle'

        return load_pickle(label_path)

    def read_activations(self, activation_name: ActivationName) -> np.array:
        l, name = activation_name
        filename = f'{name}_l{l}.pickle'

        hidden_size = None
        activations = None

        n = 0

        # The activations are stored as a series of pickle dumps, and
        # are therefore loaded until an EOFError is raised.
        with open(f'{self.activations_dir}/{filename}', 'rb') as f:
            while True:
                try:
                    sen_activations = pickle.load(f)

                    # To make hidden size dependent of data only the activations array
                    # is created only after observing the first batch of activations.
                    if hidden_size is None:
                        hidden_size = sen_activations.shape[1]
                        activations = np.zeros((self.data_len, hidden_size))

                    i = len(sen_activations)
                    activations[n:n+i] = sen_activations
                    n += i
                except EOFError:
                    break

        return activations

    def create_data_split(self,
                          activation_name: ActivationName,
                          train_subset_size: int = -1,
                          train_test_split: float = 0.9) -> DataDict:

        activations = self.read_activations(activation_name)

        split = int(self.data_len * train_test_split)

        n = split if train_subset_size == -1 else train_subset_size
        indices = np.random.choice(range(self.data_len), self.data_len, replace=False)
        train_indices = indices[:n]
        test_indices = indices[n:]

        return {
            'train_x': activations[train_indices],
            'train_y': self.labels[train_indices],
            'test_x': activations[test_indices],
            'test_y': self.labels[test_indices]
        }
