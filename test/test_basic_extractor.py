"""
Test the code in rnnalayse.extractors.base_extractor.py.
"""

import itertools
import unittest
from unittest.mock import patch, MagicMock
import os
from typing import List, Tuple, Any

import numpy as np
from overrides import overrides
import torch
from torch import Tensor

from rnnalyse.extractors.base_extractor import Extractor
from rnnalyse.models.language_model import LanguageModel
from rnnalyse.typedefs.activations import FullActivationDict, PartialActivationDict
from .test_utils import create_sentence_dummy_activations, suppress_print


# GLOBALS
ACTIVATION_DIM = 10
ACTIVATION_NAMES = [(0, "hx"), (0, "cx")]
ACTIVATIONS_DIR = "test/test_data"


class MockLanguageModel(LanguageModel):
    """
    Create a Mock version of the LanguageModel class which returns pre-defined dummy activations.
    """
    def __init__(self, num_layers: int, hidden_size: int, all_tokens: List[str], all_activations: Tensor):
        super().__init__('', '', '')
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.all_tokens = all_tokens
        self.all_activations = all_activations
        self.all_pairs = None
        self.reset()

    @overrides
    def forward(self, token: str, activations: FullActivationDict) -> Tuple[None, FullActivationDict]:
        # Consume next activation, make sure it's the right token
        next_token, next_activation = next(self.all_pairs)
        assert token == next_token

        return None, {0: {"hx": next_activation, "cx": next_activation}}

    def reset(self) -> None:
        """ Reset the activations for next test. """
        self.all_pairs = zip(self.all_tokens, self.all_activations)


class TestExtractor(unittest.TestCase):
    """ Test functionalities of the Extractor class. """

    @classmethod
    def setUpClass(cls):
        # Create directory if necessary
        if not os.path.exists(ACTIVATIONS_DIR):
            os.makedirs(ACTIVATIONS_DIR)

        # Prepare Mock sentences
        cls.test_sentences = [MagicMock(), MagicMock(), MagicMock()]
        cls.test_sentences[0].sen = ["The", "ripe", "taste", "improves", "."]
        cls.test_sentences[0].labels = [0, 0, 1, 0, 0]
        cls.test_sentences[0].misc_info = {"quality": "delicious"}

        cls.test_sentences[1].sen = ["The", "hog", "crawled", "."]
        cls.test_sentences[1].labels = [0, 1, 0, 0]
        cls.test_sentences[1].misc_info = {"quality": "hairy"}

        cls.test_sentences[2].sen = ["Move", "the", "vat", "."]
        cls.test_sentences[2].labels = [0, 0, 1, 0]
        cls.test_sentences[2].misc_info = {"quality": "ok"}

        cls.corpus = {i: cls.test_sentences[i] for i in range(len(cls.test_sentences))}

        # Mock the activations the model produces
        cls.all_tokens = list(itertools.chain(*[sentence.sen for sentence in cls.test_sentences]))
        cls.all_labels = cls._merge_labels([sentence.labels for sentence in cls.corpus.values()])

        cls.test_sentence_activations = []
        identifier_value = 0
        for sentence in cls.corpus.values():
            cls.test_sentence_activations.append(
                create_sentence_dummy_activations(len(sentence.sen), ACTIVATION_DIM, identifier_value)
            )
            identifier_value += len(sentence.sen)

        cls.all_activations = torch.cat(cls.test_sentence_activations)

        # Prepare Mock Model
        cls.model = MockLanguageModel(
            num_layers=1, hidden_size=ACTIVATION_DIM, all_tokens=cls.all_tokens, all_activations=cls.all_activations
        )

        # Init extractor
        cls.extractor = Extractor(cls.model, cls.corpus, ACTIVATION_NAMES, output_dir=ACTIVATIONS_DIR)

    @classmethod
    def tearDownClass(cls):
        # Delete activations after tests
        if os.listdir(ACTIVATIONS_DIR):
            os.remove(f"{ACTIVATIONS_DIR}/hx_l0.pickle")
            os.remove(f"{ACTIVATIONS_DIR}/cx_l0.pickle")
            os.remove(f"{ACTIVATIONS_DIR}/labels.pickle")
            os.remove(f"{ACTIVATIONS_DIR}/ranges.pickle")

    def test_extract_sentence(self):
        """ Test the _extract_sentence function for extracting the activations of whole sentences. """

        # Test extraction of all activations
        self.model.reset()
        sentences_activations, labels = zip(*[
            self.extractor._extract_sentence(sentence, lambda pos, token, sentence: True)
            for sentence in self.corpus.values()
        ])
        extracted_activations = self._merge_sentence_activations(sentences_activations)
        extracted_labels = self._merge_labels(labels)

        self.assertTrue(
            (extracted_activations == self.all_activations.numpy()).all(),
            "Selection function didn't extract all activations"
        )
        self.assertTrue(
            (extracted_labels == self.all_labels).all(),
            "Selection function didn't extract all labels"
        )

    def test_activation_extraction_by_pos(self):
        """ Test the _extract_sentence function for extracting the activations based on position. """

        self.extractor.model.reset()
        pos_sentences_activations, pos_labels = zip(*[
            self.extractor._extract_sentence(sentence, lambda pos, token, sentence: pos == 2)
            for sentence in self.corpus.values()
        ])
        extracted_pos_activations = self._merge_sentence_activations(pos_sentences_activations)
        extracted_labels = self._merge_labels(pos_labels)

        # Confirm that only one activation per sentence was extracted
        self.assertEqual(
            extracted_pos_activations.shape[0], 3, "More than one sentence was extracted based on position"
        )
        # Confirm that all extracted activations come from position 2
        # Due to the way the dummy activations are created, all their values (except for their unique id value)
        # will be 3
        self.assertTrue((extracted_pos_activations[:, 0] == 3).all(), "Sentence was extracted from the wrong position")

        # Confirm the right labels were extracted
        self.assertTrue(
            (extracted_labels == self.all_labels[[2, 7, 11]]).all(),
            "Wrong labels were extracted based on position."
        )

    def test_activation_extraction_by_label(self):
        """ Test the _extract_sentence function for extracting the activations based on label. """

        self.extractor.model.reset()
        label_sentence_activations, label_labels = zip(*[
            self.extractor._extract_sentence(sentence, lambda pos, token, sentence: sentence.labels[pos] == 1)
            for sentence in self.corpus.values()
        ])
        extracted_label_activations = self._merge_sentence_activations(label_sentence_activations)
        extracted_labels = self._merge_labels(label_labels)

        # Confirm that only one activation per sentence was extracted
        self.assertEqual(extracted_label_activations.shape[0], 3, "More than one sentence was extracted based on label")
        extracted_positions = extracted_label_activations[:, 0] - 1
        label_positions = np.array([sentence.labels.index(1) for sentence in self.test_sentences])
        # Confirm that activations are from the position of the specified label
        self.assertTrue((extracted_positions == label_positions).all(), "Wrong activations extracted based on label")

        # Confirm the right labels were extracted
        self.assertTrue(
            (extracted_labels == 1).all(),
            "Wrong labels were extracted based on label."
        )

    def test_activation_extraction_by_token(self):
        """ Test the _extract_sentence function for extracting the activations based on token. """

        self.extractor.model.reset()
        token_sentence_activations, token_labels = zip(*[
            self.extractor._extract_sentence(sentence, lambda pos, token, sentence: token == "hog")
            for sentence in self.corpus.values()
        ])
        extracted_token_activations = self._merge_sentence_activations(token_sentence_activations)
        extracted_labels = self._merge_labels(token_labels)

        # Confirm that only one activation corresponding to "hog" was extracted
        self.assertEqual(extracted_token_activations.shape[0], 1, "More than one activation extracted by token")
        assert extracted_token_activations[:, -1] == 6

        # Confirm the right labels were extracted
        self.assertTrue(
            (extracted_labels == self.all_labels[[6]]).all(),
            "Wrong labels were extracted based on token."
        )

    def test_activation_extraction_by_misc_info(self):
        """ Test the _extract_sentence function for extracting the activations based on additional info. """

        self.extractor.model.reset()
        misc_sentence_activations, misc_labels = zip(*[
            self.extractor._extract_sentence(
                sentence, lambda pos, token, sentence: sentence.misc_info["quality"] == "delicious"
            )
            for sentence in self.corpus.values()
        ])
        extracted_misc_activations = self._merge_sentence_activations(misc_sentence_activations)
        extracted_labels = self._merge_labels(misc_labels)

        # Confirm that only the first sentence was extracted
        self.assertTrue(
            (extracted_misc_activations == self.all_activations[:len(self.test_sentences[0].sen), :].numpy()).all(),
            "Wrong sentence extracted based on misc info."
        )

        # Confirm the right labels were extracted
        self.assertTrue(
            (extracted_labels == self.all_labels[:len(self.test_sentences[0].sen)]).all(),
            "Wrong labels extracted based on misc info."
        )

    # dump_pickle isn't defined in base_extractor but is imported via a from ... import ...
    # statement, therefore this patch path
    @patch('rnnalyse.extractors.base_extractor.dump_pickle')
    @suppress_print
    def test_extract_average_eos_activations(self, dump_pickle_mock: MagicMock):
        """ Test whether average end-of-sentence embeddings are calculated correctly. """

        self.extractor.model.reset()
        self.extractor.extract_average_eos_activations()
        # Get the incrementally computed activations that were used as an arg to our mock dump_pickle function
        all_avg_eos_activations, _ = dump_pickle_mock.call_args[0]

        # Confirm the the correct average eos activation was calculated
        eos_activations = [activations[-1, :].unsqueeze(0) for activations in self.test_sentence_activations]
        avg_eos_activation = torch.cat(eos_activations, dim=0).mean(dim=0)
        self.assertTrue(
            (avg_eos_activation == all_avg_eos_activations[0]["hx"]).all(),
            "Average end of sentence activations have wrong value."
        )

        self.assertEqual(
            len(all_avg_eos_activations[0]["hx"]), self.extractor.model.hidden_size,
            "Average end of sentence activations have wrong dimensions."
        )

    @suppress_print
    @patch('rnnalyse.activations.activation_writer.ActivationWriter.dump_activations')
    def test_extraction_dumping_args(self, dump_activations_mock: MagicMock):
        """
        Test whether functions used to dump pickle files during activation extraction are called
        with the right arguments.
        """

        self.extractor.model.reset()
        self.extractor.extract()
        call_arg = dump_activations_mock.call_args[0][0]

        # Validate function calls
        self.assertEqual(dump_activations_mock.call_count, 3, "Function was called the wrong number of times.")
        self.assertTrue(
            self.is_partial_activation_dict(call_arg),
            "Function was called with wrong type of variable, expected PartialActivationDict."
        )

    @patch('rnnalyse.extractors.base_extractor.dump_pickle')
    @suppress_print
    def test_average_eos_dumping_args(self, dump_pickle_mock: MagicMock):
        """
        Test whether functions used to dump pickle files during thge calculation of the average end-of-sentence
        activations are called with the right arguments.
        """
        self.extractor.model.reset()
        self.extractor.extract_average_eos_activations()
        first_arg, second_arg = dump_pickle_mock.call_args[0]

        # Validate function call
        dump_pickle_mock.assert_called_once()
        self.assertTrue(
            self.is_full_activation_dict(first_arg),
            "First argument of dump function received a wrong type of argument, excepted FullActivationDict."
        )

        self.assertEqual(
            second_arg, f"{ACTIVATIONS_DIR}/avg_eos.pickle", "Second argument is not a string or the wrong path."
        )

    @staticmethod
    def _merge_sentence_activations(sentences_activations: List[PartialActivationDict]) -> np.array:
        """ Merge activations from different sentences into one single numpy array. """
        return np.array(list(itertools.chain(
            *[sentence_activations[(0, "hx")] for sentence_activations in sentences_activations])
        ))

    @staticmethod
    def _merge_labels(sentence_labels: List[np.array]):
        """ Merge labels from different sentences into a single numpy array. """
        return np.array(list(itertools.chain(*sentence_labels)))

    @staticmethod
    def is_full_activation_dict(var: Any) -> bool:
        """ Check whether a variable is of type FullActivationDict. """

        # This way of checking the type is rather awkward, but there seems to be no function to compare a variable
        # against a subscripted generic - believe me, I also hate this
        first_outer_key = list(var.keys())[0]
        first_value = list(var.values())[0]
        first_inner_key = list(var[first_outer_key].keys())[0]

        return all([
            type(var) == dict,
            type(first_outer_key) == int,
            type(first_value) == dict,
            type(var[first_outer_key][first_inner_key]) == Tensor
        ])

    @staticmethod
    def is_partial_activation_dict(var: Any) -> bool:
        """ Check whether a variable is of type PartialActivationDict. """
        first_key = list(var.keys())[0]
        first_value = list(var.values())[0]

        return all([
            type(var) == dict,
            type(first_key) == tuple,
            type(first_key[0]) == int,
            type(first_key[1]) == str,
            type(first_value) in (np.array, np.ndarray)
        ])
