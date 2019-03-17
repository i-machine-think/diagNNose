from collections import defaultdict
from time import time
from typing import Any, List, Optional

import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegressionCV as LogReg
from sklearn.metrics import accuracy_score

from rnnalyse.typedefs.activations import ActivationName
from rnnalyse.typedefs.classifiers import ResultsDict
from rnnalyse.activations.activation_reader import ActivationReader
from rnnalyse.utils.paths import dump_pickle, trim


class DCTrainer:
    """ Trains Diagnostic Classifiers (DC) on extracted activation data.

    For each activation that is part of the provided activation_names
    argument a different classifier will be trained.

    Parameters
    ----------
    activations_dir : str
        Path to folder containing the activations to train on.
    activation_names : List[ActivationName]
        List of (layer, name)-tuples indicating which activations the
        classifiers will be trained on.
    output_dir : str
        Path to folder to which models and results will be saved.
    classifier_type : str
        Classifier type, right now only accepts 'logreg' or 'svm'.
    label_path : str, optional
        Path to label files. If not provided, labels.pickle in
        `activations_dir` will be used.
    use_class_weights : bool
        Flag to indicate whether class weights calculated from the
        training data should be used for training (defaults to True).

    Attributes
    ----------
    activation_reader : ActivationReader
        Class that reads and preprocesses activation data.
    classifier : Classifier
        Current classifier that is being trained.
    results : ResultsDict
        Dictionary containing relevant results. TODO: Add preds to this instead of separate files?
    """
    def __init__(self,
                 activations_dir: str,
                 activation_names: List[ActivationName],
                 output_dir: str,
                 classifier_type: str,
                 label_path: Optional[str] = None,
                 use_class_weights: bool = True) -> None:

        self.activation_names: List[ActivationName] = activation_names
        self.output_dir = trim(output_dir)
        self.classifier_type = classifier_type
        # TODO: Allow own classifier here (should adhere to some base functions, such as .fit())
        self.use_class_weights = use_class_weights

        self.activation_reader = ActivationReader(activations_dir, label_path)
        self._reset_classifier()
        self.results: ResultsDict = defaultdict(dict)

    def train(self, train_subset_size: int = -1, train_test_split: float = 0.9) -> None:
        start_t = time()

        for a_name in self.activation_names:
            data_dict = self.activation_reader.create_data_split(a_name,
                                                                 train_subset_size,
                                                                 train_test_split)

            # Calculate class weights
            if self.use_class_weights:
                classes, class_freqs = np.unique(data_dict['train_y'], return_counts=True)
                norm = class_freqs.sum()  # Norm factor
                class_weight = {classes[i]: class_freqs[i] / norm for i in range(len(class_freqs))}  # Normalize
                self.classifier.class_weight = class_weight

            # Train
            self.fit_data(data_dict['train_x'], data_dict['train_y'], a_name)
            pred_y = self.eval_classifier(data_dict['test_x'], data_dict['test_y'], a_name)

            self.save_classifier(pred_y, a_name)
            self._reset_classifier()

        self.log_results(start_t)

    def _reset_classifier(self) -> None:
        self.classifier = {
            'logreg': LogReg(),
            'svm': None,
        }[self.classifier_type]

    def fit_data(self,
                 train_x: np.ndarray, train_y: np.ndarray,
                 activation_name: ActivationName) -> None:
        print(f'\nStarting fitting model on {activation_name}...')

        start_time = time()
        self.classifier.fit(train_x, train_y)

        print(f'Fitting done in {time() - start_time:.2f}s')

    # TODO: Add more evaluation metrics here
    def eval_classifier(self,
                        test_x: np.ndarray, test_y: np.ndarray,
                        activation_name: ActivationName) -> np.ndarray:
        pred_y = self.classifier.predict(test_x)

        acc = accuracy_score(test_y, pred_y)

        print(f'{activation_name} acc.:', acc)

        self.results[activation_name]['acc'] = acc

        return pred_y

    def save_classifier(self, pred_y: np.ndarray, activation_name: ActivationName) -> None:
        l, name = activation_name

        preds_path = f'{self.output_dir}/preds/{name}_l{l}.pickle'
        model_path = f'{self.output_dir}/models/{name}_l{l}.joblib'

        dump_pickle(pred_y, preds_path)
        joblib.dump(self.classifier, model_path)

    @staticmethod
    def load_classifier(path: str) -> Any:
        return joblib.load(path)

    def log_results(self, start_t: float) -> None:
        total_time = time() - start_t
        print(f'Total classification time took {total_time:.2f}s')

        log = {
            'activation_names': self.activation_names,
            'classifier_type': self.classifier_type,
            'results': self.results,
            'total_time': total_time,
        }

        log_path = f'{self.output_dir}/log.pickle'
        dump_pickle(log, log_path)
