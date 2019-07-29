"""
Test the code in rnnalayse.activations.activation_reader.py.
"""
import os
import unittest
from typing import List, Sequence

from torch import Tensor

from diagnnose.activations.activation_reader import ActivationReader

from .test_utils import create_and_dump_dummy_activations

# GLOBALS
ACTIVATIONS_DIM = 10
ACTIVATIONS_DIR = "test/test_data"
ACTIVATIONS_NAME = "hx_l0"
NUM_TEST_SENTENCES = 5


class TestActivationReader(unittest.TestCase):
    """ Test functionalities of the ActivationReader class. """

    @classmethod
    def setUpClass(cls) -> None:
        # Create directory if necessary
        if not os.path.exists(ACTIVATIONS_DIR):
            os.makedirs(ACTIVATIONS_DIR)

        create_and_dump_dummy_activations(
            num_sentences=NUM_TEST_SENTENCES,
            activations_dim=ACTIVATIONS_DIM,
            max_tokens=5,
            activations_dir=ACTIVATIONS_DIR,
            activations_name=ACTIVATIONS_NAME,
            num_classes=2,
        )
        cls.activation_reader = ActivationReader(activations_dir=ACTIVATIONS_DIR)

    @classmethod
    def tearDownClass(cls) -> None:
        # Remove files from previous tests
        if os.listdir(ACTIVATIONS_DIR):
            os.remove(f"{ACTIVATIONS_DIR}/{ACTIVATIONS_NAME}.pickle")
            os.remove(f"{ACTIVATIONS_DIR}/labels.pickle")
            os.remove(f"{ACTIVATIONS_DIR}/ranges.pickle")

    def test_read_activations(self) -> None:
        """ Test reading activations from a pickle file. """
        activations = self.activation_reader.read_activations((0, "hx"))

        # Check if the amount of read data is correct
        self.assertEqual(
            self.activation_reader.data_len,
            activations.shape[0],
            "Number of read activations is wrong.",
        )

        # Check how many sentences were processed
        # The first activation of a dummy sentence is a vector of ones
        start_of_sentences = activations[:, 0] == 1
        num_read_sentences = start_of_sentences.sum()

        self.assertEqual(
            NUM_TEST_SENTENCES, num_read_sentences, "Number of read sentences is wrong"
        )

    def test_activation_indexing(self) -> None:
        first_idx = list(self.activation_reader.activation_ranges.keys())[0]

        def seq_shapes(tensors: Sequence[Tensor]) -> List[Sequence[int]]:
            shapes = map(lambda t: tuple(t.shape), tensors)
            return list(shapes)

        self.assertEqual(
            seq_shapes(
                self.activation_reader[0, {"indextype": "pos", "a_name": (0, "hx")}]
            ),
            seq_shapes(
                self.activation_reader[
                    first_idx, {"indextype": "key", "a_name": (0, "hx")}
                ]
            ),
            "Activation shape of first sentence not equal by position/key indexing",
        )
        self.assertEqual(
            seq_shapes(self.activation_reader[0:]),
            seq_shapes(
                self.activation_reader[slice(0, None, None), {"indextype": "key"}]
            ),
            "Indexing all activations by key and position yields different results",
        )
        self.assertEqual(
            seq_shapes(self.activation_reader[0]),
            seq_shapes(self.activation_reader[first_idx, {"indextype": "key"}]),
            "Activation shape of first sentence not equal by position/key indexing",
        )
        data_len = self.activation_reader.data_len
        ashape = self.activation_reader[
            slice(0, data_len // 2, None), {"indextype": "all"}
        ][0].shape
        self.assertTrue(
            ashape == (data_len // 2, ACTIVATIONS_DIM),
            f"Indexing by all activations is not working: {ashape}",
        )

    def test_activation_ranges(self) -> None:
        self.assertEqual(
            sum(
                ma - mi for mi, ma in self.activation_reader.activation_ranges.values()
            ),
            self.activation_reader.data_len,
            "Length mismatch activation ranges and label length of ActivationReader",
        )
