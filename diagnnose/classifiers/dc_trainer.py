import os
from time import time
from typing import Any, Dict, Optional, Tuple

import sklearn.metrics as metrics
import torch
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegressionCV
from skorch import NeuralNetClassifier
from torch import Tensor

from diagnnose.activations.data_loader import DataLoader
from diagnnose.extractors.simple_extract import simple_extract
from diagnnose.typedefs.activations import (
    ActivationName,
    ActivationNames,
    SelectionFunc,
)
from diagnnose.typedefs.classifiers import ControlTask, DataDict
from diagnnose.typedefs.corpus import Corpus
from diagnnose.typedefs.models import LanguageModel
from diagnnose.utils.pickle import dump_pickle

from .logreg import LogRegModule


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
    test_selection_func : SelectFunc, optional
        Selection function that determines whether a corpus item should
        be taken into account for testing. If such a function has been
        used during extraction, make sure to pass it along here as well.
    classifier_type : str, optional
        Either `logreg_torch`, using a torch logreg model, or
        `logreg_sklearn`, using a LogisticRegressionCV model of sklearn.
    control_task : ControlTask, optional
        Control task function of Hewitt et al. (2019), mapping a corpus
        item to a random label. If not provided the corpus labels will
        be used instead.
    verbose : int, optional
        Set to any positive number for verbosity. Defaults to 0.

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
        selection_func: SelectionFunc = lambda sen_id, pos, example: True,
        test_selection_func: Optional[SelectionFunc] = None,
        control_task: Optional[ControlTask] = None,
        classifier_type: str = "logreg_torch",
        verbose: int = 0,
    ) -> None:
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.remove_callbacks = []
        activations_dir, test_activations_dir = self._extract_activations(
            save_dir,
            corpus,
            activation_names,
            selection_func,
            activations_dir,
            test_activations_dir,
            test_corpus,
            test_selection_func,
            model,
        )

        self.model = model
        self.activation_names = activation_names
        self.data_dict: DataDict = {}
        self.data_loader = DataLoader(
            activations_dir,
            corpus,
            test_activations_dir=test_activations_dir,
            test_corpus=test_corpus,
            selection_func=selection_func,
            test_selection_func=test_selection_func,
            control_task=control_task,
        )
        assert classifier_type in [
            "logreg_torch",
            "logreg_sklearn",
        ], "Classifier type not understood, should be either `logreg_toch` or `logreg_sklearn`"
        self.classifier_type = classifier_type
        self.verbose = verbose

    def train(
        self,
        calc_class_weights: bool = False,
        data_subset_size: int = -1,
        train_test_split: float = 0.9,
        store_activations: bool = True,
        rank: Optional[int] = None,
    ) -> Dict[ActivationName, Any]:
        """ Trains DCs on multiple activation names.

        Parameters
        ----------
        calc_class_weights : bool, optional
            Set to True to calculate the classifier class weights based on
            the corpus class frequencies. Defaults to False.
        data_subset_size : int, optional
            Size of the subset on which training will be performed.
            Defaults to the full set of activations.
        train_test_split : float, optional
            Percentage of the train/test split. If separate test
            activations are provided this split won't be used.
            Defaults to 0.9/0.1.
        store_activations : bool, optional
            Set to True to store the extracted activations. Defaults to
            True.
        rank : int, optional
            Matrix rank of the linear classifier. Defaults to the full
            rank if not provided.
        """
        full_results_dict = {}

        for activation_name in self.activation_names:
            results_dict = self._train(
                activation_name,
                calc_class_weights=calc_class_weights,
                data_subset_size=data_subset_size,
                train_test_split=train_test_split,
                rank=rank,
            )
            full_results_dict[activation_name] = results_dict

        if not store_activations:
            for remove_callback in self.remove_callbacks:
                remove_callback()

        return full_results_dict

    def _train(
        self,
        activation_name: ActivationName,
        calc_class_weights: bool = False,
        data_subset_size: int = -1,
        train_test_split: float = 0.9,
        rank: Optional[int] = None,
    ) -> Dict[str, Any]:
        """ Initiates training the DC on 1 activation type. """
        self.data_dict = self.data_loader.create_data_split(
            activation_name, data_subset_size, train_test_split
        )

        self._reset_classifier(rank=rank)
        if self.verbose > 0:
            train_size = self.data_dict["train_x"].size(0)
            test_size = self.data_dict["test_x"].size(0)
            print(f"train/test: {train_size}/{test_size}")

        # Calculate class weights
        if calc_class_weights:
            self._set_class_weights(self.data_dict["train_y"])

        # Train
        self._fit(activation_name)
        results_dict = self._eval(self.data_dict["test_y"])

        self._save_classifier(activation_name)

        if self.data_dict["train_y_control"] is not None:
            self._control_task(rank, results_dict)

        self._save_results(results_dict, activation_name)

        return results_dict

    def _fit(self, activation_name: ActivationName) -> None:
        start_time = time()
        if self.verbose > 0:
            print(f"\nStarting fitting model on {activation_name}...")

        self.classifier.fit(self.data_dict["train_x"], self.data_dict["train_y"])

        if self.verbose > 0:
            print(f"Fitting done in {time() - start_time:.2f}s")

    def _eval(self, labels: Tensor) -> Dict[str, Any]:
        pred_y = self.classifier.predict(self.data_dict["test_x"])

        acc = metrics.accuracy_score(labels, pred_y)
        f1 = metrics.f1_score(labels, pred_y, average="micro")
        mcc = metrics.matthews_corrcoef(labels, pred_y)
        cm = metrics.confusion_matrix(labels, pred_y)

        # TODO: remove later
        # proba = self.classifier.predict_proba(self.data_dict["test_x"])
        logits = self.classifier.infer(self.data_dict["test_x"], create_softmax=False)

        results_dict = {
            "probs": logits.detach(),
            "accuracy": acc,
            "f1": f1,
            "mcc": mcc,
            "confusion_matrix": cm,
        }

        return results_dict

    def _control_task(self, rank: Optional[int], results_dict: Dict[str, Any]) -> None:
        if self.verbose > 0:
            print("Starting fitting the control task...")
        self._reset_classifier(rank=rank)
        self.classifier.fit(
            self.data_dict["train_x"], self.data_dict["train_y_control"]
        )

        results_dict_control = self._eval(self.data_dict["test_y_control"])
        for k, v in results_dict_control.items():
            results_dict[f"{k}_control"] = v
        results_dict["selectivity"] = (
            results_dict["accuracy"] - results_dict["accuracy_control"]
        )

    def _save_classifier(self, activation_name: ActivationName):
        if self.save_dir is not None:
            l, name = activation_name
            model_path = os.path.join(self.save_dir, f"{name}_l{l}.joblib")
            joblib.dump(self.classifier, model_path)

    def _save_results(
        self, results_dict: Dict[str, Any], activation_name: ActivationName
    ) -> None:
        if self.verbose > 0:
            for k, v in results_dict.items():
                print(k, v, "", sep="\n")
            print("Label vocab:", self.data_loader.label_vocab.itos)

        if self.save_dir is not None:
            l, name = activation_name
            preds_path = os.path.join(self.save_dir, f"{name}_l{l}_results.pickle")
            dump_pickle(results_dict, preds_path)

    def _reset_classifier(self, rank: Optional[int] = None) -> None:
        if self.classifier_type == "logreg_torch":
            ninp = self.data_dict["train_x"].size(1)
            nout = len(self.data_loader.label_vocab)
            self.classifier = NeuralNetClassifier(
                LogRegModule(ninp=ninp, nout=nout, rank=rank),
                lr=0.01,
                max_epochs=3,
                verbose=self.verbose,
                optimizer=torch.optim.Adam,
            )
        elif self.classifier_type == "logreg_sklearn":
            self.classifier = LogisticRegressionCV()

    # TODO: comply with skorch
    def _set_class_weights(self, labels: Tensor) -> None:
        classes, class_freqs = torch.unique(labels, return_counts=True)
        norm = class_freqs.sum().item()
        class_weight = {
            classes[i].item(): class_freqs[i].item() / norm
            for i in range(len(class_freqs))
        }
        self.classifier.class_weight = class_weight

    def _extract_activations(
        self,
        save_dir: str,
        corpus: Corpus,
        activation_names: ActivationNames,
        selection_func: SelectionFunc,
        activations_dir: Optional[str],
        test_activations_dir: Optional[str],
        test_corpus: Optional[Corpus],
        test_selection_func: Optional[SelectionFunc],
        model: Optional[LanguageModel],
    ) -> Tuple[str, Optional[str]]:
        if activations_dir is None:
            # We combine the 2 selection funcs to extract train and test activations simultaneously.
            if test_corpus is None and test_selection_func is not None:

                def new_selection_func(idx, pos, item):
                    return selection_func(idx, pos, item) or test_selection_func(
                        idx, pos, item
                    )

            else:
                new_selection_func = selection_func

            activations_dir = os.path.join(save_dir, "activations")
            remove_callback = simple_extract(
                model, activations_dir, corpus, activation_names, new_selection_func
            )
            self.remove_callbacks.append(remove_callback)

        if test_corpus is not None and test_activations_dir is None:
            test_activations_dir = os.path.join(save_dir, "test_activations")
            remove_callback = simple_extract(
                model,
                test_activations_dir,
                test_corpus,
                activation_names,
                test_selection_func or (lambda sen_id, pos, example: True),
            )
            self.remove_callbacks.append(remove_callback)

        return activations_dir, test_activations_dir
