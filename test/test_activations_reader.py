"""
Test the code in rnnalayse.activations.activations_reader.py.
"""
import pickle
import unittest
import os
import random
import torch

from rnnalyse.activations.activations_reader import ActivationsReader


# GLOBALS
ACTIVATIONS_DIR = "test_data"
ACTIVATIONS_NAME = "hx_l0"
NUM_TEST_SENTENCES = 5
TRAIN_TEST_SPLIT = 0.75


class TestActivationsReader(unittest.TestCase):
    """ Test functionalities of the ActivationsReader class. """

    @classmethod
    def setUpClass(self):
        # Remove files from previous tests
        if os.listdir(ACTIVATIONS_DIR):
            os.remove(f"{ACTIVATIONS_DIR}/{ACTIVATIONS_NAME}.pickle")
            os.remove(f"{ACTIVATIONS_DIR}/labels.pickle")

        # Create dummy data have reader read it
        self.num_labels = self.create_dummy_activations(
            num_sentences=NUM_TEST_SENTENCES, activations_dim=10, max_tokens=5, activations_dir=ACTIVATIONS_DIR,
            activations_name=ACTIVATIONS_NAME
        )
        self.activation_reader = ActivationsReader(activations_dir=ACTIVATIONS_DIR)

    def test_read_activations(self):
        activations = self.activation_reader.read_activations((0, "hx"))
        labels = self.activation_reader.labels

        # Check if the amount of read data is correct
        assert self.num_labels == labels.shape[0] == activations.shape[0]

        # Check how many sentences were processed
        start_of_sentences = activations[:, 0] == 1  # The first activation of a dummy sentence is a vector of ones
        num_read_sentences = start_of_sentences.astype(int).sum()

        assert NUM_TEST_SENTENCES == num_read_sentences

    def test_create_data_split(self):
        data_dict = self.activation_reader.create_data_split((0, "hx"), train_test_split=TRAIN_TEST_SPLIT)
        train_x, train_y = data_dict["train_x"], data_dict["train_y"]
        test_x, test_y = data_dict["test_x"], data_dict["test_y"]

        # Test the data set lengths
        assert train_x.shape[0] == train_y.shape[0] == int(self.num_labels * TRAIN_TEST_SPLIT)
        assert test_x.shape[0] == test_y.shape[0] == self.num_labels - int(self.num_labels * TRAIN_TEST_SPLIT)

        # Test whether training and test set are disjoint
        # Use the identifier values that sit on the last dimension of each activation
        train_ids = set(train_x[:, -1])
        test_ids = set(test_x[:, -1])

        assert len(train_ids & test_ids) == 0

        # TODO: Test additional arguments once we agree on them

    @staticmethod
    def create_dummy_activations(num_sentences: int, activations_dim: int, max_tokens: int, activations_dir: str,
                                 activations_name: str):
        with open(f"{activations_dir}/{activations_name}.pickle", "wb") as f:
            num_labels = 0
            activation_identifier = 0  # Identify activations globally by adding on number on one end

            for i in range(num_sentences):
                num_activations = random.randint(1, max_tokens)  # Determine the number of tokens in this sentence
                num_labels += num_activations
                activations = torch.ones(num_activations, activations_dim)

                # First activation is a vector of ones, second activation a vector of twos and so on
                # (except for an extra identifier dimension which will come in handy later)
                for n in range(num_activations-1):
                    activations[n+1:, :] += 1

                # Add identifier value for each activation
                identifier_values = torch.arange(activation_identifier, activation_identifier + num_activations)
                activations[:, -1] = identifier_values  # Add values on last dimension of activation
                activation_identifier += num_activations  # Increment global activation id

                pickle.dump(activations, f)

        # Generate some random labels and dump them
        with open(f"{activations_dir}/labels.pickle", "wb") as f:
            labels = torch.rand(num_labels)
            labels[labels > 0.5] = 1
            labels[labels <= 0.5] = 0
            pickle.dump(labels, f)

        return num_labels

