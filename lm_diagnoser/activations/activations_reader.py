import pickle

import numpy as np

from ..typedefs.models import ActivationName
from ..typedefs.classifiers import DataDict


class ActivationsReader:
    def __init__(self,
                 activations_dir: str,
                 labels: str = 'labels.pickle',
                 keys: str = 'keys.pickle') -> None:

        self.activations_dir = activations_dir

        self.keys = self._read_keys(keys)
        self.labels = self._read_labels(labels)
        self.data_len = len(self.labels)

    def _read_keys(self, keys_name: str) -> np.array:
        with open(f'{self.activations_dir}/{keys_name}', 'rb') as f:
            keys = pickle.load(f)

        return keys

    def _read_labels(self, label_name: str) -> np.array:
        with open(f'{self.activations_dir}/{label_name}', 'rb') as f:
            labels = pickle.load(f)

        return labels

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
                          subset_size: int = -1,
                          train_split: float = 0.9) -> DataDict:

        activations = self.read_activations(activation_name)

        split = int(self.data_len * train_split)

        n = split if subset_size == -1 else subset_size
        train_indices = np.random.choice(range(n), n, replace=False)

        return {
            'train_x': activations[train_indices],
            'train_y': self.labels[train_indices],
            'test_x': activations[split:],
            'test_y': self.labels[split:]
        }
