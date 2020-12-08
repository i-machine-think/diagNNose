import os
import warnings
from time import time
from typing import Any, Dict, Optional

import torch
from torch import Tensor

from diagnnose.typedefs.activations import ActivationName
from diagnnose.typedefs.probe import DataDict, DCConfig
from diagnnose.utils.pickle import dump_pickle

from .data_loader import DataLoader
from .logreg import L1NeuralNetClassifier, LogRegModule


class DCTrainer:
    """Trains Diagnostic Classifiers (DC) on extracted activation data.

    For each activation that is part of the provided activation_names
    argument a different classifier will be trained.

    Parameters
    ----------
    data_loader : DataLoader
        ``DataLoader`` that contains the activations and labels on
        which the DCs will be trained. A ``DataLoader`` can contain
        activations for multiple layers and gates of a model, for which
        separate DCs will be trained and evaluated.
    save_dir : str
        Directory to which trained models will be saved.
    lr : float, optional
        Learning rate of the linear classifier that is used during
        training. Defaults to 0.01.
    max_epochs : int, optional
        Maximum number of training epochs used for cross-validation.
        Defaults to 10.
    rank : int, optional
        Matrix rank of the linear classifier. Defaults to the full rank
        if not provided.
    lambda1 : float, optional
        Coefficient for L1 regularization that can be increased to
        induce sparsity in the diagnostic classifier. Defaults to 0.,
        indicating no L1 regularization.
    verbose : int, optional
        Set to any positive number for verbosity. Defaults to 0.

    Attributes
    ----------
    classifier : Classifier
        Current classifier that is being trained.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        save_dir: str,
        lr: float = 0.01,
        max_epochs: int = 10,
        rank: Optional[int] = None,
        lambda1: float = 0.0,
        verbose: int = 0,
    ) -> None:
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.classifier = None

        self.data_loader = data_loader

        self.dc_config = DCConfig(lr, max_epochs, rank, lambda1, verbose)

    def train(self) -> Dict[ActivationName, Any]:
        """Trains DCs on multiple activation names."""

        full_results_dict = {}

        for activation_name in self.data_loader.activation_names:
            results_dict = self._train_one_dc(activation_name)
            full_results_dict[activation_name] = results_dict

        return full_results_dict

    def _train_one_dc(self, activation_name: ActivationName) -> Dict[str, Any]:
        """ Initiates training the DC on 1 activation type. """
        data_dict: DataDict = self.data_loader.load(activation_name)

        if self.dc_config.verbose > 0:
            train_size = data_dict.train_activations.size(0)
            test_size = data_dict.test_activations.size(0)
            print(f"train/test: {train_size}/{test_size}")
            print(f"\nStarting fitting model on {activation_name}...")

        # Train
        self._fit(data_dict.train_activations, data_dict.train_labels)

        results_dict = self._eval(data_dict.test_activations, data_dict.test_labels)

        self._save_classifier(activation_name)

        if data_dict.train_control_labels is not None:
            self._fit(data_dict.train_activations, data_dict.train_control_labels)
            control_results = self._eval(
                data_dict.test_activations, data_dict.test_control_labels
            )
            results_dict["control"] = control_results
            results_dict["selectivity"] = (
                results_dict["accuracy"] - control_results["accuracy"]
            )
            self._save_classifier(activation_name, postfix="_control")

        self._save_results(results_dict, activation_name)

        return results_dict

    def _fit(self, activations: Tensor, labels: Tensor) -> None:
        self._reset_classifier(activations.size(1), len(torch.unique(labels)))

        start_time = time()

        self.classifier.fit(activations, labels)

        if self.dc_config.verbose > 0:
            print(f"Fitting done in {time() - start_time:.2f}s")

    def _eval(self, activations: Tensor, labels: Tensor) -> Dict[str, Any]:
        try:
            import sklearn.metrics as metrics
        except ImportError:
            warnings.warn("sklearn.metrics is needed for DC evaluation")
            raise

        pred_y = self.classifier.predict(activations)

        acc = metrics.accuracy_score(labels, pred_y)
        f1 = metrics.f1_score(labels, pred_y, average="micro")
        mcc = metrics.matthews_corrcoef(labels, pred_y)
        cm = metrics.confusion_matrix(labels, pred_y)

        results_dict = {"accuracy": acc, "f1": f1, "mcc": mcc, "confusion_matrix": cm}

        return results_dict

    def _save_classifier(self, activation_name: ActivationName, postfix: str = ""):
        if self.save_dir is not None:
            l, name = activation_name
            fn = f"{name}_l{l}" + postfix
            model_path = os.path.join(self.save_dir, fn + ".pt")
            torch.save(self.classifier.module.state_dict(), model_path)

    def _save_results(
        self, results_dict: Dict[str, Any], activation_name: ActivationName
    ) -> None:
        if self.dc_config.verbose > 0:
            for k, v in results_dict.items():
                print(k, v, "", sep="\n")
            print("Label vocab:", self.data_loader.label_vocab)

        if self.save_dir is not None:
            l, name = activation_name
            results_path = os.path.join(self.save_dir, f"{name}_l{l}_results.pickle")
            dump_pickle(results_dict, results_path)

    def _reset_classifier(self, ninp: int, nout: int) -> None:
        dc_config_dict = self.dc_config._asdict()
        rank = dc_config_dict.pop("rank")

        self.classifier = L1NeuralNetClassifier(
            LogRegModule(ninp=ninp, nout=nout, rank=rank),
            optimizer=torch.optim.Adam,
            **dc_config_dict,
        )
