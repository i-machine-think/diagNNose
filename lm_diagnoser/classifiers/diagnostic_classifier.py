import pickle
from collections import defaultdict
from time import time
from typing import Any, Dict, List

import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegressionCV as LogReg
from sklearn.metrics import accuracy_score

from ..typedefs.models import ActivationName


class DiagnosticClassifier:
    def __init__(self,
                 activations_dir: str,
                 activation_names: List[ActivationName],
                 output_dir: str,
                 labels: str = 'labels.pickle',
                 keys: str = 'keys.pickle') -> None:

        self.activations_dir = activations_dir
        self.activation_names: List[ActivationName] = activation_names
        self.output_dir = output_dir

        self.keys = self._read_keys(keys)
        self.labels = self._read_labels(labels)
        self.data_len = len(self.labels)

    # TODO: Write json file to ouput path containing experiment information
    def classify(self, subset_size: int = -1) -> None:

        start_time = time()

        results: Dict[ActivationName, List[float]] = defaultdict(list)

        for activation_name in self.activation_names:
            l, name = activation_name
            activations = self._read_activations(activation_name)

            data_dict = self._split_data(activations, subset_size=subset_size)

            model = LogReg()

            print(f'\nStarting fitting model on {activation_name}...')
            t0 = time()

            model.fit(data_dict['train_x'], data_dict['train_y'])

            print(f'Fitting done in {time() - t0:.2f}s')
            y_pred = model.predict(data_dict['test_x'])
            acc = accuracy_score(data_dict['test_y'], y_pred)
            print(f'{activation_name} acc.:', acc)
            results[activation_name].append(acc)

            with open(f'{self.output_dir}/preds/{name}_l{l}.pickle', 'wb') as file:
                pickle.dump(y_pred, file)
            joblib.dump(model, f'{self.output_dir}/models/{name}_l{l}.pickle')

        print(f'Total classification time took {time() - start_time:.2f}s')

        for k, v in results.items():
            print(f'{k} {np.mean(v):.4f}+/-{np.std(v):.6f}')

    def _read_keys(self, keys_name: str) -> np.array:
        with open(f'{self.activations_dir}/{keys_name}', 'rb') as f:
            keys = pickle.load(f)

        return keys

    def _read_labels(self, label_name: str) -> np.array:
        with open(f'{self.activations_dir}/{label_name}', 'rb') as f:
            labels = pickle.load(f)

        return labels

    def _read_activations(self, activation_name: ActivationName) -> np.array:
        l, name = activation_name
        filename = f'{name}_l{l}.pickle'

        hidden_size = None
        activations = None

        n = 0

        with open(f'{self.activations_dir}/{filename}', 'rb') as f:
            while True:
                try:
                    sen_activations = pickle.load(f)

                    if hidden_size is None:
                        hidden_size = sen_activations.shape[1]
                        activations = np.zeros((self.data_len, hidden_size))

                    i = len(sen_activations)
                    activations[n:n+i] = sen_activations
                    n += i
                except EOFError:
                    break

        return activations

    def _split_data(self,
                    activations: np.array,
                    subset_size: int = -1,
                    train_split: float = 0.9) -> Any:

        split = int(self.data_len * train_split)

        n = split if subset_size == -1 else subset_size
        train_indices = np.random.choice(range(n), n, replace=False)

        return {
            'train_x': activations[train_indices],
            'train_y': self.labels[train_indices],
            'test_x': activations[split:],
            'test_y': self.labels[split:]
        }
