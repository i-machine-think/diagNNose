"""
Test the code in rnnalayse.activations.activations_reader.py.
"""
import unittest
import os

from rnnalyse.activations.activations_reader import ActivationsReader
from .test_utils import create_and_dump_dummy_activations

# GLOBALS
ACTIVATIONS_DIR = "test/test_data"
ACTIVATIONS_NAME = "hx_l0"
NUM_TEST_SENTENCES = 5
TRAIN_TEST_SPLIT = 0.75

if not os.path.exists(ACTIVATIONS_DIR):
    os.makedirs(ACTIVATIONS_DIR)


class TestActivationsReader(unittest.TestCase):
    """ Test functionalities of the ActivationsReader class. """

    @classmethod
    def setUpClass(self):
        # Remove files from previous tests
        if os.listdir(ACTIVATIONS_DIR):
            os.remove(f"{ACTIVATIONS_DIR}/{ACTIVATIONS_NAME}.pickle")
            os.remove(f"{ACTIVATIONS_DIR}/labels.pickle")

        # Create dummy data have reader read it
        labels = create_and_dump_dummy_activations(
            num_sentences=NUM_TEST_SENTENCES, activations_dim=10, max_tokens=5, activations_dir=ACTIVATIONS_DIR,
            activations_name=ACTIVATIONS_NAME, num_classes=2
        )
        self.num_labels = labels.shape[0]
        self.activation_reader = ActivationsReader(activations_dir=ACTIVATIONS_DIR)

    def test_read_activations(self):
        activations = self.activation_reader.read_activations((0, "hx"))
        labels = self.activation_reader.labels

        # Check if the amount of read data is correct
        self.assertEqual(self.num_labels, labels.shape[0], "Number of read labels is wrong.")
        self.assertEqual(self.num_labels, activations.shape[0], "Number of read activations is wrong.")

        # Check how many sentences were processed
        start_of_sentences = activations[:, 0] == 1  # The first activation of a dummy sentence is a vector of ones
        num_read_sentences = start_of_sentences.astype(int).sum()

        self.assertEqual(NUM_TEST_SENTENCES, num_read_sentences, "Number of read sentences is wrong")

    def test_create_data_split(self):
        data_dict = self.activation_reader.create_data_split((0, "hx"), train_test_split=TRAIN_TEST_SPLIT)
        train_x, train_y = data_dict["train_x"], data_dict["train_y"]
        test_x, test_y = data_dict["test_x"], data_dict["test_y"]

        # Test the data set lengths
        right_num_train = int(self.num_labels * TRAIN_TEST_SPLIT)
        right_num_test = self.num_labels - int(self.num_labels * TRAIN_TEST_SPLIT)
        self.assertEqual(train_x.shape[0], right_num_train, "Wrong size of training split.")
        self.assertEqual(train_y.shape[0], right_num_train, "Wrong size of training split.")
        self.assertEqual(test_x.shape[0], right_num_test, "Wrong size of test split.")
        self.assertEqual(test_y.shape[0], right_num_test, "Wrong size of test split.")

        # Test whether training and test set are disjoint
        # Use the identifier values that sit on the last dimension of each activation
        train_ids = set(train_x[:, -1])
        test_ids = set(test_x[:, -1])

        self.assertEqual(len(train_ids & test_ids), 0, "Training and test set are not disjoint!")

        # TODO: Test additional arguments once we agree on them

