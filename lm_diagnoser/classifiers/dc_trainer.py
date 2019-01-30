import pickle
from collections import defaultdict
from time import time
from typing import Any, List, Optional

import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegressionCV as LogReg
from sklearn.metrics import accuracy_score

from ..typedefs.models import ActivationName
from ..typedefs.classifiers import ResultsDict
from ..activations.activations_reader import ActivationsReader


class DiagnosticClassifier:
    def __init__(self,
                 activations_dir: str,
                 activation_names: List[ActivationName],
                 output_dir: str,
                 classifier: str,
                 label_path: Optional[str] = None) -> None:

        self.activation_names: List[ActivationName] = activation_names
        self.output_dir = output_dir
        self.classifier = classifier

        self.activations_reader = ActivationsReader(activations_dir, label_path)
        self.model: Any = None
        self.results: ResultsDict = defaultdict(dict)

    def classify(self, subset_size: int = -1) -> None:
        start_t = time()

        for a_name in self.activation_names:
            self._create_classifier()
            data_dict = self.activations_reader.create_data_split(a_name, subset_size)

            self.fit_data(data_dict['train_x'], data_dict['train_y'], a_name)
            pred_y = self.eval_model(data_dict['test_x'], data_dict['test_y'], a_name)

            self.save_model(pred_y, a_name)

        self.log_results(start_t)

    def _create_classifier(self) -> None:
        self.model = {
            'logreg': LogReg(),
            'svm': None,
        }[self.classifier]

    def fit_data(self,
                 train_x: np.array, train_y: np.array,
                 activation_name: ActivationName) -> None:
        print(f'\nStarting fitting model on {activation_name}...')

        start_time = time()
        self.model.fit(train_x, train_y)

        print(f'Fitting done in {time() - start_time:.2f}s')

    # TODO: Add more evaluation metrics here
    def eval_model(self,
                   test_x: np.array, test_y: np.array,
                   activation_name: ActivationName) -> np.ndarray:
        pred_y = self.model.predict(test_x)

        acc = accuracy_score(test_y, pred_y)

        print(f'{activation_name} acc.:', acc)

        self.results[activation_name]['acc'] = acc

        return pred_y

    def save_model(self, pred_y: np.ndarray, activation_name: ActivationName) -> None:
        l, name = activation_name

        with open(f'{self.output_dir}/preds/{name}_l{l}.pickle', 'wb') as file:
            pickle.dump(pred_y, file)

        joblib.dump(self.model, f'{self.output_dir}/models/{name}_l{l}.pickle')

    def log_results(self, start_t: float) -> None:
        total_time = time() - start_t
        print(f'Total classification time took {total_time:.2f}s')

        log = {
            'activation_names': self.activation_names,
            'classifier': self.classifier,
            'results': self.results,
            'total_time': total_time,
        }

        with open(f'{self.output_dir}/log.pickle', 'wb') as file:
            pickle.dump(log, file)
