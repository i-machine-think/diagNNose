import os
from collections import defaultdict
from time import time
from typing import Any, List

import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegressionCV as LogRegCV
from sklearn.metrics import accuracy_score

from diagnnose.activations.data_loader import DataLoader
from diagnnose.typedefs.activations import ActivationName
from diagnnose.typedefs.classifiers import ResultsDict
from diagnnose.utils.paths import dump_pickle
from diagnnose.typedefs.corpus import Corpus


class DCTrainer:
    """ Trains Diagnostic Classifiers (DC) on extracted activation data.

    For each activation that is part of the provided activation_names
    argument a different classifier will be trained.

    Parameters
    ----------
    corpus : Corpus
        Corpus containing the token labels for each sentence.
    activations_dir : str
        Path to folder containing the activations to train on.
    activation_names : List[ActivationName]
        List of activation names on which classifiers will be trained.
    save_dir : str
        Directory to which trained models will be saved.
    classifier_type : str
        Classifier type, as of now only accepts `logreg`, but more will be added.
    calc_class_weights : bool, optional
        Set to True to calculate the classifier class weights based on
        the corpus class frequencies. Defaults to False.

    Attributes
    ----------
    data_loader : DataLoader
        Class that reads and preprocesses activation data.
    classifier : Classifier
        Current classifier that is being trained.
    results : ResultsDict
        Dictionary containing relevant results. TODO: Add preds to this instead of separate files?
    """
    def __init__(self,
                 corpus: Corpus,
                 activations_dir: str,
                 activation_names: List[ActivationName],
                 save_dir: str,
                 classifier_type: str,
                 calc_class_weights: bool = False) -> None:

        self.activation_names: List[ActivationName] = activation_names
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # TODO: Allow own classifier here (should adhere to some base functions, such as .fit())
        self.classifier_type = classifier_type
        self.calc_class_weights = calc_class_weights

        self.data_loader = DataLoader(activations_dir, corpus)
        self.results: ResultsDict = defaultdict(dict)

        self._reset_classifier()

    def train(self, data_subset_size: int = -1, train_test_split: float = 0.9) -> None:
        start_t = time()

        for a_name in self.activation_names:
            data_dict = self.data_loader.create_data_split(a_name,
                                                           data_subset_size,
                                                           train_test_split)

            # Calculate class weights
            if self.calc_class_weights:
                classes, class_freqs = np.unique(data_dict['train_y'], return_counts=True)
                norm = class_freqs.sum()  # Norm factor
                class_weight = {classes[i]: class_freqs[i] / norm for i in range(len(class_freqs))}
                self.classifier.class_weight = class_weight

            # Train
            self.fit_data(data_dict['train_x'], data_dict['train_y'], a_name)
            pred_y = self.eval_classifier(data_dict['test_x'], data_dict['test_y'], a_name)

            self.save_classifier(pred_y, a_name)
            self._reset_classifier()

        self.log_results(start_t)

    def _reset_classifier(self) -> None:
        self.classifier = {
            'logreg': LogRegCV(),
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

        preds_path = os.path.join(self.save_dir, f'{name}_l{l}_preds.pickle')
        model_path = os.path.join(self.save_dir, f'{name}_l{l}.joblib')

        dump_pickle(pred_y, preds_path)
        joblib.dump(self.classifier, model_path)

    @staticmethod
    def load_classifier(path: str) -> Any:
        return joblib.load(path)

    def log_results(self, start_t: float) -> None:
        total_time = time() - start_t
        m, s = divmod(total_time, 60)

        print(f'Total classification time took {m:.0f}m {s:.1f}s')

        log = {
            'activation_names': self.activation_names,
            'classifier_type': self.classifier_type,
            'results': self.results,
            'total_time': total_time,
        }

        log_path = os.path.join(self.save_dir, 'log.pickle')
        dump_pickle(log, log_path)
