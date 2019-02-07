"""
Test the code in rnnalayse.classifiers.dc_trainer.py.
"""

from collections import Counter
import unittest
from unittest.mock import patch
import os

from rnnalyse.classifiers.dc_trainer import DCTrainer
from .test_utils import create_and_dump_dummy_activations

# GLOBALS
ACTIVATIONS_DIR = "test_data"
ACTIVATION_NAMES = [(0, "hx")]
ACTIVATIONS_NAME = "hx_l0"
NUM_TEST_SENTENCES = 5


class TestDCTrainer(unittest.TestCase):
    """ Test functionalities of the DCTrainer class. """

    @classmethod
    def setUpClass(self):
        # Remove files from previous tests
        if os.listdir(ACTIVATIONS_DIR):
            os.remove(f"{ACTIVATIONS_DIR}/{ACTIVATIONS_NAME}.pickle")
            os.remove(f"{ACTIVATIONS_DIR}/labels.pickle")

        # Create dummy data have reader read it
        self.labels = create_and_dump_dummy_activations(
            num_sentences=NUM_TEST_SENTENCES, activations_dim=10, max_tokens=7, activations_dir=ACTIVATIONS_DIR,
            activations_name=ACTIVATIONS_NAME, num_classes=5
        )

        # Model without class weights
        self.model = DCTrainer(
            activations_dir=ACTIVATIONS_DIR , activation_names=ACTIVATION_NAMES, output_dir=ACTIVATIONS_DIR,
            classifier_type="logreg", label_path=f"{ACTIVATIONS_DIR}/labels.pickle", use_class_weights=False
        )
        # Model with class weights
        self.weighed_model = DCTrainer(
            activations_dir=ACTIVATIONS_DIR, activation_names=ACTIVATION_NAMES, output_dir=ACTIVATIONS_DIR,
            classifier_type="logreg", label_path=f"{ACTIVATIONS_DIR}/labels.pickle"
        )
        # Create split here s.t. we can later mock this exact function in DCTrainer.train
        # This way we can use the same random data splits and make sure class weights are counted correctly,
        # otherwise this variable would be inside the local scope of the function and inaccessible
        self.data_dict = self.weighed_model.activations_reader.create_data_split(ACTIVATION_NAMES[0])

    @patch('rnnalyse.activations.activations_reader.ActivationsReader.create_data_split')
    @patch('rnnalyse.classifiers.dc_trainer.DCTrainer._reset_classifier')
    @patch('rnnalyse.classifiers.dc_trainer.DCTrainer.log_results')
    @patch('rnnalyse.classifiers.dc_trainer.DCTrainer.save_classifier')
    @patch('rnnalyse.classifiers.dc_trainer.DCTrainer.eval_classifier')
    @patch('rnnalyse.classifiers.dc_trainer.DCTrainer.fit_data')
    def test_class_weights(self, mock_fit_data, mock_eval_classifier, mock_save_classifier, mock_log_results,
                           mock_reset_classifier, create_data_split_mock):
        create_data_split_mock.return_value = self.data_dict

        # Confirm that class weights are not used if flag is not given
        mock_eval_classifier.return_value = self.labels  # Fake predictions
        self.model.train()
        assert self.model.classifier.class_weight is None

        # Confirm that class weights are calculated correctly if actually used
        class_counts = Counter(self.data_dict["train_y"].numpy())
        num_labels = sum(class_counts.values())
        self.weighed_model.train()
        assert all([
            class_counts[class_] / num_labels == weight
            for class_, weight in self.weighed_model.classifier.class_weight.items()
        ])

