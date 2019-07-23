"""
Test the code in rnnalayse.classifiers.dc_trainer.py.
"""

import os
import unittest
from collections import Counter
from unittest.mock import MagicMock, patch

from diagnnose.classifiers.dc_trainer import DCTrainer

from .test_utils import create_and_dump_dummy_activations

# GLOBALS
ACTIVATION_NAMES = [(0, "hx")]
ACTIVATIONS_NAME = "hx_l0"
NUM_TEST_SENTENCES = 5
ACTIVATIONS_DIR = "test/test_data"


class TestDCTrainer(unittest.TestCase):
    """ Test functionalities of the DCTrainer class. """

    @classmethod
    def setUpClass(cls) -> None:
        # Create directory if necessary
        if not os.path.exists(ACTIVATIONS_DIR):
            os.makedirs(ACTIVATIONS_DIR)

        # Create dummy data have reader read it
        cls.labels = create_and_dump_dummy_activations(
            num_sentences=NUM_TEST_SENTENCES,
            activations_dim=10,
            max_tokens=7,
            activations_dir=ACTIVATIONS_DIR,
            activations_name=ACTIVATIONS_NAME,
            num_classes=5,
        )

        # Model without class weights
        cls.model = DCTrainer(
            ACTIVATIONS_DIR,
            ACTIVATION_NAMES,
            ACTIVATIONS_DIR,
            "logreg",
            labels=cls.labels,
            calc_class_weights=False,
        )
        # Model with class weights
        cls.weighed_model = DCTrainer(
            ACTIVATIONS_DIR,
            ACTIVATION_NAMES,
            ACTIVATIONS_DIR,
            "logreg",
            labels=cls.labels,
            calc_class_weights=True,
        )
        # Create split here s.t. we can later mock this exact function in DCTrainer.train
        # This way we can use the same random data splits
        # and make sure class weights are counted correctly,
        # otherwise this variable would be inside the local scope of the function and inaccessible
        cls.data_dict = cls.weighed_model.data_loader.create_data_split(
            ACTIVATION_NAMES[0]
        )

    @classmethod
    def tearDownClass(cls) -> None:
        # Remove files after tests
        if os.listdir(ACTIVATIONS_DIR):
            os.remove(f"{ACTIVATIONS_DIR}/{ACTIVATIONS_NAME}.pickle")
            os.remove(f"{ACTIVATIONS_DIR}/ranges.pickle")
            os.remove(f"{ACTIVATIONS_DIR}/labels.pickle")

    @patch("diagnnose.activations.data_loader.DataLoader.create_data_split")
    @patch("diagnnose.classifiers.dc_trainer.DCTrainer._reset_classifier")
    @patch("diagnnose.classifiers.dc_trainer.DCTrainer.log_results")
    @patch("diagnnose.classifiers.dc_trainer.DCTrainer.save_classifier")
    @patch("diagnnose.classifiers.dc_trainer.DCTrainer.eval_classifier")
    @patch("diagnnose.classifiers.dc_trainer.DCTrainer.fit_data")
    def test_class_weights(
        self,
        _mock_fit_data: MagicMock,
        mock_eval_classifier: MagicMock,
        _mock_save_classifier: MagicMock,
        _mock_log_results: MagicMock,
        _mock_reset_classifier: MagicMock,
        create_data_split_mock: MagicMock,
    ) -> None:
        create_data_split_mock.return_value = self.data_dict

        # Confirm that class weights are not used if flag is not given
        mock_eval_classifier.return_value = self.labels  # Fake predictions
        self.model.train()
        self.assertIsNone(
            self.model.classifier.class_weight,
            "Class weights are given although flag is set to False",
        )

        # Confirm that class weights are calculated correctly if actually used
        class_counts = Counter(self.data_dict["train_y"].numpy())
        num_labels = sum(class_counts.values())
        self.weighed_model.train()
        self.assertTrue(
            all(
                [
                    class_counts[class_] / num_labels == weight
                    for class_, weight in self.weighed_model.classifier.class_weight.items()
                ]
            ),
            "Class weights have wrong values.",
        )
