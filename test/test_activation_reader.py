"""
Test the code in rnnalayse.activations.activation_reader.py.
"""
import unittest
import random
import os

from rnnalyse.activations.activation_reader import ActivationReader
from .test_utils import create_and_dump_dummy_activations

# GLOBALS
ACTIVATIONS_DIR = "test/test_data"
ACTIVATIONS_NAME = "hx_l0"
NUM_TEST_SENTENCES = 5


class TestActivationReader(unittest.TestCase):
    """ Test functionalities of the ActivationReader class. """

    @classmethod
    def setUpClass(cls):
        # Create directory if necessary
        if not os.path.exists(ACTIVATIONS_DIR):
            os.makedirs(ACTIVATIONS_DIR)

        # Create dummy data have reader read it
        labels = create_and_dump_dummy_activations(
            num_sentences=NUM_TEST_SENTENCES, activations_dim=10, max_tokens=5, activations_dir=ACTIVATIONS_DIR,
            activations_name=ACTIVATIONS_NAME, num_classes=2
        )
        cls.num_labels = labels.shape[0]
        cls.activation_reader = ActivationReader(activations_dir=ACTIVATIONS_DIR)

    @classmethod
    def tearDownClass(cls):
        # Remove files from previous tests
        if os.listdir(ACTIVATIONS_DIR):
            os.remove(f"{ACTIVATIONS_DIR}/{ACTIVATIONS_NAME}.pickle")
            os.remove(f"{ACTIVATIONS_DIR}/labels.pickle")
            os.remove(f"{ACTIVATIONS_DIR}/ranges.pickle")

    def test_read_activations(self):
        """ Test reading activations from a pickle file. """
        activations = self.activation_reader.read_activations((0, "hx"))
        labels = self.activation_reader.labels

        # Check if the amount of read data is correct
        self.assertEqual(self.num_labels, labels.shape[0], "Number of read labels is wrong.")
        self.assertEqual(self.num_labels, activations.shape[0], "Number of read activations is wrong.")

        # Check how many sentences were processed
        start_of_sentences = activations[:, 0] == 1  # The first activation of a dummy sentence is a vector of ones
        num_read_sentences = start_of_sentences.astype(int).sum()

        self.assertEqual(NUM_TEST_SENTENCES, num_read_sentences, "Number of read sentences is wrong")

    def test_activation_indexing(self):
        self.activation_reader.activations = (0, 'hx')
        self.assertEqual(
            self.activation_reader[0:].shape,
            self.activation_reader.get_by_sen_key(slice(0, None, None)).shape,
            'Indexing all activations by key and position yields different results'
        )
        first_index = list(self.activation_reader.activation_ranges.keys())[0]
        self.assertEqual(
            self.activation_reader[0].shape,
            self.activation_reader.get_by_sen_key(first_index).shape,
            'Activation shape of first sentence not equal by position/key indexing'
        )

    def test_activation_ranges(self):
        self.assertEqual(
            sum(ma-mi for mi, ma in self.activation_reader.activation_ranges.values()),
            self.activation_reader.data_len,
            'Length mismatch activation ranges and label length of ActivationReader'
        )

    def test_create_data_split(self):
        """ Test creating the data set splits for Diagnostic Classifier training. """

        # Validate data splits for the full data set
        train_test_split = random.uniform(0.1, 0.9)
        full_data_dict = self.activation_reader.create_data_split((0, "hx"), train_test_split=train_test_split)
        train_x, train_y = full_data_dict["train_x"], full_data_dict["train_y"]
        test_x, test_y = full_data_dict["test_x"], full_data_dict["test_y"]
        self._validate_data_split(train_x, train_y, test_x, test_y, size=self.num_labels, data_split=train_test_split)

        # Validate data splits for a partial data set
        cutoff = random.randrange(5, self.num_labels - 1)
        partial_data_dict = self.activation_reader.create_data_split(
            (0, "hx"), train_test_split=train_test_split, data_subset_size=cutoff
        )
        partial_train_x, partial_train_y = partial_data_dict["train_x"], partial_data_dict["train_y"]
        partial_test_x, partial_test_y = partial_data_dict["test_x"], partial_data_dict["test_y"]
        self._validate_data_split(
            partial_train_x, partial_train_y, partial_test_x, partial_test_y,
            size=cutoff, data_split=train_test_split
        )

    def _validate_data_split(self, train_x, train_y, test_x, test_y, size, data_split):
        """ Validate size and content of an arbitrary data split. """
        # Test the data set lengths
        right_num_train = int(size * data_split)
        right_num_test = size - int(size * data_split)
        self.assertEqual(train_x.shape[0], right_num_train, "Wrong size of training split.")
        self.assertEqual(train_y.shape[0], right_num_train, "Wrong size of training split.")
        self.assertEqual(test_x.shape[0], right_num_test, "Wrong size of test split.")
        self.assertEqual(test_y.shape[0], right_num_test, "Wrong size of test split.")

        # Test whether training and test set are disjoint
        # Use the identifier values that sit on the last dimension of each activation
        train_ids = set(train_x[:, -1])
        test_ids = set(test_x[:, -1])

        self.assertEqual(len(train_ids & test_ids), 0, "Training and test set are not disjoint!")

