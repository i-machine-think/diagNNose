import os
from time import time
from typing import Any, Dict, Optional, Tuple

import torch
from diagnnose.activations.data_loader import DataLoader
from diagnnose.typedefs.activations import ActivationName, ActivationNames, SelectFunc
from diagnnose.typedefs.corpus import Corpus
from diagnnose.utils.pickle import dump_pickle
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegressionCV as LogRegCV
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import Tensor
from diagnnose.extractors.simple_extract import simple_extract
from diagnnose.models.lm import LanguageModel


class DCTrainer:
    """ Trains Diagnostic Classifiers (DC) on extracted activation data.

    For each activation that is part of the provided activation_names
    argument a different classifier will be trained.

    Parameters
    ----------
    save_dir : str, optional
        Directory to which trained models will be saved, if provided.
    corpus : Corpus
        Corpus containing the token labels for each sentence.
    activation_names : List[ActivationName]
        List of activation names on which classifiers will be trained.
    activations_dir : str, optional
        Path to folder containing the activations to train on. If not
        provided newly extracted activations will be saved to
        `save_dir`.
    test_activations_dir : str, optional
        Directory containing the extracted test activations. If not
        provided the train activation set will be split and partially
        used as test set.
    test_corpus : Corpus, optional
        Corpus containing the test labels for each sentence. If
        provided without `test_activations_dir` newly extracted
        activations will be saved to `save_dir`.
    model : LanguageModel, optional
        LanguageModel that should be provided if new activations need
        to be extracted prior to training the classifiers.
    selection_func : SelectFunc, optional
        Selection function that determines whether a corpus item should
        be taken into account for training. If such a function has been
        used during extraction, make sure to pass it along here as well.

    Attributes
    ----------
    data_loader : DataLoader
        Class that reads and preprocesses activation data.
    classifier : Classifier
        Current classifier that is being trained.
    """

    def __init__(
        self,
        save_dir: str,
        corpus: Corpus,
        activation_names: ActivationNames,
        activations_dir: Optional[str] = None,
        test_activations_dir: Optional[str] = None,
        test_corpus: Optional[Corpus] = None,
        model: Optional[LanguageModel] = None,
        selection_func: SelectFunc = lambda sen_id, pos, example: True,
    ) -> None:
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        activations_dir, test_activations_dir = self._extract_activations(
            save_dir,
            corpus,
            activation_names,
            selection_func,
            activations_dir,
            test_activations_dir,
            test_corpus,
            model,
        )

        self.activation_names = activation_names
        self.data_loader = DataLoader(
            activations_dir,
            corpus,
            test_activations_dir=test_activations_dir,
            test_corpus=test_corpus,
            selection_func=selection_func,
        )
        self.classifier = LogRegCV()

    def train(
        self,
        calc_class_weights: bool = False,
        data_subset_size: int = -1,
        train_test_split: float = 0.9,
    ) -> None:
        """ Trains DCs on multiple activation names.

        Parameters
        ----------
        calc_class_weights : bool, optional
            Set to True to calculate the classifier class weights based on
            the corpus class frequencies. Defaults to False.
        data_subset_size : int, optional
            Size of the subset on which training will be performed. Defaults
            to the full set of activations.
        train_test_split : float, optional
            Percentage of the train/test split. If separate test
            activations are provided this split won't be used.
            Defaults to 0.9/0.1.
        """
        for activation_name in self.activation_names:
            self._train(
                activation_name,
                calc_class_weights=calc_class_weights,
                data_subset_size=data_subset_size,
                train_test_split=train_test_split,
            )

    def _train(
        self,
        activation_name: ActivationName,
        calc_class_weights: bool = False,
        data_subset_size: int = -1,
        train_test_split: float = 0.9,
    ) -> None:
        """ Initiates training the DC on 1 activation type. """
        self._reset_classifier()

        data_dict = self.data_loader.create_data_split(
            activation_name, data_subset_size, train_test_split
        )

        # Calculate class weights
        if calc_class_weights:
            self._set_class_weights(data_dict["train_y"])

        # Train
        self._fit(data_dict["train_x"], data_dict["train_y"], activation_name)
        results = self._eval(data_dict["test_x"], data_dict["test_y"])

        if self.save_dir is not None:
            self._save(results, activation_name)

    def _fit(
        self, train_x: Tensor, train_y: Tensor, activation_name: ActivationName
    ) -> None:
        start_time = time()
        print(f"\nStarting fitting model on {activation_name}...")

        self.classifier.fit(train_x, train_y)

        print(f"Fitting done in {time() - start_time:.2f}s")

    def _eval(self, test_x: Tensor, test_y: Tensor) -> Dict[str, Any]:
        pred_y = self.classifier.predict(test_x)

        acc = accuracy_score(test_y, pred_y)
        cm = confusion_matrix(test_y, pred_y)

        results = {"accuracy": acc, "confusion matrix": cm}
        for k, v in results.items():
            print(k, v, "", sep="\n")
        results["pred_y"] = pred_y

        return results

    def _save(self, results: Dict[str, Any], activation_name: ActivationName) -> None:
        l, name = activation_name

        preds_path = os.path.join(self.save_dir, f"{name}_l{l}_results.pickle")
        model_path = os.path.join(self.save_dir, f"{name}_l{l}.joblib")

        dump_pickle(results, preds_path)
        joblib.dump(self.classifier, model_path)

    def _reset_classifier(self) -> None:
        self.classifier = LogRegCV()

    def _set_class_weights(self, train_y: Tensor) -> None:
        classes, class_freqs = torch.unique(train_y, return_counts=True)
        norm = class_freqs.sum().item()
        class_weight = {
            classes[i].item(): class_freqs[i].item() / norm
            for i in range(len(class_freqs))
        }
        self.classifier.class_weight = class_weight

    @staticmethod
    def _extract_activations(
        save_dir: str,
        corpus: Corpus,
        activation_names: ActivationNames,
        selection_func: SelectFunc,
        activations_dir: Optional[str],
        test_activations_dir: Optional[str],
        test_corpus: Optional[Corpus],
        model: Optional[LanguageModel],
    ) -> Tuple[str, Optional[str]]:
        if activations_dir is None:
            activations_dir = os.path.join(save_dir, "activations")
            simple_extract(
                model, activations_dir, corpus, activation_names, selection_func
            )

        if test_corpus is not None and test_activations_dir is None:
            test_activations_dir = os.path.join(save_dir, "test_activations")
            simple_extract(
                model,
                test_activations_dir,
                test_corpus,
                activation_names,
                selection_func,
            )

        return activations_dir, test_activations_dir
