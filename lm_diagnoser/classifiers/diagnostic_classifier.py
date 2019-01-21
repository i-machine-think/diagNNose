import pickle
from time import time
from typing import Any, List

import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegressionCV as LogReg
from sklearn.metrics import accuracy_score

from typedefs.models import ActivationName

OUTPUT_PATH = 'classifiers/trained'


class DiagnosticClassifier:
    def __init__(self,
                 embedding_location: str,
                 activation_names: List[ActivationName],
                 hidden_size: int) -> None:

        self.embedding_location = embedding_location
        self.activation_names: List[ActivationName] = activation_names
        self.hidden_size = hidden_size

        self.keys = self._read_keys()
        self.labels = self._read_labels()
        self.data_len = len(self.labels)

    @staticmethod
    def load(path):
        return joblib.load(path)

    def classify(self) -> None:
        for activation_name in self.activation_names:
            l, name = activation_name
            activations = self._read_activations(activation_name)

            data_dict = self._split_data(activations)

            model = LogReg()

            print(f'Starting fitting model on {activation_name}...')
            t0 = time()

            model.fit(data_dict['train_x'], data_dict['train_y'])

            print('Fitting done in', time() - t0)
            y_pred = model.predict(data_dict['test_x'])
            print(f'{activation_name} acc.:', accuracy_score(data_dict['test_y'], y_pred))

            with open(f'{OUTPUT_PATH}/preds-{name}_l{l}.pickle', 'wb') as file:
                pickle.dump(y_pred, file)
            joblib.dump(model, f'{OUTPUT_PATH}/lr-{name}_l{l}.pickle')

    def _read_keys(self) -> Any:
        with open(f'{self.embedding_location}/keys.pickle', 'rb') as f:
            keys = pickle.load(f)

        return keys

    def _read_labels(self) -> Any:
        with open(f'{self.embedding_location}/labels.pickle', 'rb') as f:
            labels = pickle.load(f)

        return labels

    def _read_activations(self, activation_name: ActivationName) -> np.array:
        l, name = activation_name
        filename = f'{name}_l{l}.pickle'
        activations = np.zeros((self.data_len, self.hidden_size))

        n = 0

        with open(f'{self.embedding_location}/{filename}', 'rb') as f:
            while True:
                try:
                    sen_activations = pickle.load(f)
                    i = len(sen_activations)
                    activations[n:n+i] = sen_activations
                    n += i
                except EOFError:
                    break

        return activations

    def _split_data(self, activations: np.array, train_split: float = 0.9) -> Any:
        n = self.data_len
        perm_i = np.random.choice(range(n), n, replace=False)

        x_data = activations[perm_i]
        y_data = self.labels[perm_i]

        split = int(train_split * n)

        return {
            'train_x': x_data[:split],
            'train_y': y_data[:split],
            'test_x': x_data[split:],
            'test_y': y_data[split:]
        }

# def _create_model(self, model_type, TRAIN_Y=None, kernel='rbf', probs=True, balance=False, balance_param=0.5, model_path='') -> None:
#     npi_counts = Counter(TRAIN_Y)
#     if balance:
#         assert TRAIN_Y is not None, 'Provide label data for class balancing'
#         npi_balance = {x: len(TRAIN_Y) / c ** balance_param for x, c in npi_counts.items()}
#         print(npi_balance)
#     else:
#         npi_balance = {x: 1 for x in npi_counts.keys()}
#
#     if model_path:
#         model = joblib.load(model_path)
#         print('Pretrained model loaded...')
#     else:
#         model = {
#             'svm': SVC(kernel=kernel, decision_function_shape='ovo', class_weight=npi_balance,
#                        verbose=0, probability=probs),
#             'logreg': LogReg(),
#         }[model_type]
#
#     return model
#
#
# def classify(x, y, x_test, y_test, model, sample_size=-1, save_model=True):
#     if sample_size > 0:
#         sample_i = np.random.choice(range(len(y)), sample_size, replace=False)
#         x = x[sample_i]
#         y = y[sample_i]
#
#     print('Starting fitting model...')
#     t0 = time()
#
#     model.fit(x, y)
#
#     print('Fitting done in', time() - t0)
#     y_pred = model.predict(x_test)
#     print('Accuracy:', accuracy_score(y_test, y_pred))
#
#     date = datetime.now().strftime("%d-%m|%H:%M")
#
#     if save_model:
#         with open(output_path + 'preds-' + date + '.pickle', 'wb') as file:
#             pickle.dump(y_pred, file)
#         joblib.dump(model, output_path + 'lr-' + date + '.pickle')
#         print('Saved at:', output_path + 'lr-' + date + '.pickle')
#         date = datetime.now().strftime("%d-%m|%H:%M")
#
#
# def run_experiments(xpathname, ypathname, sample_size=-1, model_type='logreg', kernel='rbf', save_model=True, kill=False):
#     print('Start experiment:', model_type, sample_size, datetime.now().strftime("%d-%m|%H:%M"))
#
#     TRAIN_X, TEST_X, TRAIN_Y, TEST_Y, perm_i = get_data(xpathname, ypathname)
#
#     model = create_model('logreg')
#
#     classify(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, model, sample_size=sample_size, save_model=save_model)
#
#     if kill:
#         subprocess.call(["systemctl", "suspend"])
