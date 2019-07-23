"""
Test the code in rnnalayse.activations.activation_reader.py.
"""
import os
import random
import unittest

from diagnnose.activations.data_loader import DataLoader
from diagnnose.typedefs.classifiers import DataDict

from .test_utils import create_and_dump_dummy_activations

# GLOBALS
ACTIVATIONS_DIM = 10
ACTIVATIONS_DIR = "test/test_data"
ACTIVATIONS_NAME = "hx_l0"
NUM_TEST_SENTENCES = 5


class TestDataLoader(unittest.TestCase):
    """ Test functionalities of the ActivationReader class. """

    @classmethod
    def setUpClass(cls) -> None:
        # Create directory if necessary
        if not os.path.exists(ACTIVATIONS_DIR):
            os.makedirs(ACTIVATIONS_DIR)

        # Create dummy data have reader read it
        labels = create_and_dump_dummy_activations(
            num_sentences=NUM_TEST_SENTENCES,
            activations_dim=ACTIVATIONS_DIM,
            max_tokens=5,
            activations_dir=ACTIVATIONS_DIR,
            activations_name=ACTIVATIONS_NAME,
            num_classes=2,
        )

        cls.data_loader = DataLoader(ACTIVATIONS_DIR, labels=labels)
        cls.num_labels = cls.data_loader.data_len

    @classmethod
    def tearDownClass(cls) -> None:
        # Remove files from previous tests
        if os.listdir(ACTIVATIONS_DIR):
            os.remove(f"{ACTIVATIONS_DIR}/{ACTIVATIONS_NAME}.pickle")
            os.remove(f"{ACTIVATIONS_DIR}/ranges.pickle")
            os.remove(f"{ACTIVATIONS_DIR}/labels.pickle")

    def test_create_data_split(self) -> None:
        """ Test creating the data set splits for Diagnostic Classifier training. """

        # Validate data splits for the full data set
        train_test_split = random.uniform(0.1, 0.9)
        full_data_dict = self.data_loader.create_data_split(
            (0, "hx"), train_test_split=train_test_split
        )
        self._validate_data_split(full_data_dict, self.num_labels, train_test_split)

        # Validate data splits for a partial data set
        cutoff = random.randrange(5, self.num_labels - 1)
        partial_data_dict = self.data_loader.create_data_split(
            (0, "hx"), train_test_split=train_test_split, data_subset_size=cutoff
        )

        self._validate_data_split(
            partial_data_dict, size=cutoff, data_split=train_test_split
        )

    def _validate_data_split(
        self, full_data_dict: DataDict, size: int, data_split: float
    ) -> None:
        """ Validate size and content of an arbitrary data split. """
        train_x, train_y = full_data_dict["train_x"], full_data_dict["train_y"]
        test_x, test_y = full_data_dict["test_x"], full_data_dict["test_y"]

        right_num_train = int(size * data_split)
        right_num_test = size - int(size * data_split)

        self.assertEqual(
            train_x.shape[0], right_num_train, "Wrong size of training split."
        )
        self.assertEqual(
            train_y.shape[0], right_num_train, "Wrong size of training split."
        )
        self.assertEqual(test_x.shape[0], right_num_test, "Wrong size of test split.")
        self.assertEqual(test_y.shape[0], right_num_test, "Wrong size of test split.")

        # Test whether training and test set are disjoint
        # Use the identifier values that sit on the last dimension of each activation
        train_ids = set(train_x[:, -1])
        test_ids = set(test_x[:, -1])

        self.assertEqual(
            len(train_ids & test_ids), 0, "Training and test set are not disjoint!"
        )
