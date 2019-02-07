"""
Test the code in rnnalayse.extractors.base_extractor.py.
"""

import itertools
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from overrides import overrides
import torch

from rnnalyse.extractors.base_extractor import Extractor
from rnnalyse.models.language_model import LanguageModel
from test_utils import create_sentence_dummy_activations


# GLOBALS
ACTIVATION_DIM = 10
ACTIVATION_NAMES = [(0, "hx"), (0, "cx")]
ACTIVATIONS_DIR = "test_data"


class MockLanguageModel(LanguageModel):
    """
    Create a Mock version of the LanguageModel class which returns pre-defined dummy activations.
    """
    def __init__(self, num_layers, hidden_size, all_tokens, all_activations):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.all_tokens = all_tokens
        self.all_activations = all_activations
        self.all_pairs = None
        self.reset()

    @overrides
    def forward(self, token, activations):
        # Consume next activation, make sure it's the right token
        next_token, next_activation = next(self.all_pairs)
        assert token == next_token

        return None, {0: {"hx": next_activation, "cx": next_activation}}

    def reset(self):
        """ Reset the activations for next test. """
        self.all_pairs = zip(self.all_tokens, self.all_activations)


class TestExtractor(unittest.TestCase):
    """ Test functionalities of the Extractor class. """

    @classmethod
    def setUpClass(self):
        # Prepare Mock sentences
        self.test_sentences = [MagicMock(), MagicMock(), MagicMock()]
        self.test_sentences[0].sen = ["The", "ripe", "taste", "improves", "."]
        self.test_sentences[0].labels = [0, 0, 1, 0, 0]
        self.test_sentences[0].misc_info = {"quality": "delicious"}

        self.test_sentences[1].sen = ["The", "hog", "crawled", "."]
        self.test_sentences[1].labels = [0, 1, 0, 0]
        self.test_sentences[1].misc_info = {"quality": "hairy"}

        self.test_sentences[2].sen = ["Move", "the", "vat", "."]
        self.test_sentences[2].labels = [0, 0, 1, 0]
        self.test_sentences[2].misc_info = {"quality": "ok"}

        self.corpus = {i: self.test_sentences[i] for i in range(len(self.test_sentences))}

        # Mock the activations the model produces
        self.all_tokens = list(itertools.chain(*[sentence.sen for sentence in self.test_sentences]))

        self.test_sentence_activations = []
        identifier_value = 0
        for sentence in self.corpus.values():
            self.test_sentence_activations.append(
                create_sentence_dummy_activations(len(sentence.sen), ACTIVATION_DIM, identifier_value)
            )
            identifier_value += len(sentence.sen)

        self.all_activations = torch.cat(self.test_sentence_activations)

        # Prepare Mock Model
        self.model = MockLanguageModel(
            num_layers=1, hidden_size=ACTIVATION_DIM, all_tokens=self.all_tokens, all_activations=self.all_activations
        )

        # Init extractor
        self.extractor = Extractor(self.model, self.corpus, ACTIVATION_NAMES, output_dir=ACTIVATIONS_DIR)

    def test_extract_sentence(self):
        """ Test the _extract_sentence class, especially with focus on the selection function functionality. """

        def _merge_sentence_activations(sentences_activations):
            return np.array(list(itertools.chain(
                *[sentence_activations[(0, "hx")] for sentence_activations in sentences_activations])
            ))

        # Test extraction of all activations
        self.model.reset()
        sentences_activations = [
            self.extractor._extract_sentence(sentence, lambda pos, token, sentence: True)
            for sentence in self.corpus.values()
        ]
        extracted_activations = _merge_sentence_activations(sentences_activations)
        assert (extracted_activations == self.all_activations.numpy()).all()

        # Test extraction selection by position
        self.extractor.model.reset()
        pos_sentences_activations = [
            self.extractor._extract_sentence(sentence, lambda pos, token, sentence: pos == 2)
            for sentence in self.corpus.values()
        ]
        extracted_pos_activations = _merge_sentence_activations(pos_sentences_activations)
        assert extracted_pos_activations.shape[0] == 3  # Confirm that only one activation per sentence was extracted
        # Confirm that all extracted activations come from position 2
        # Due to the way the dummy activations are created, all their values (except for their unique id value)
        # will be 3
        assert (extracted_pos_activations[:, 0] == 3).all()

        # Test extraction selection by label
        self.extractor.model.reset()
        label_sentence_activations = [
            self.extractor._extract_sentence(sentence, lambda pos, token, sentence: sentence.labels[pos] == 1)
            for sentence in self.corpus.values()
        ]
        extracted_label_activations = _merge_sentence_activations(label_sentence_activations)
        assert extracted_label_activations.shape[0] == 3  # Confirm that only one activation per sentence was extracted
        extracted_positions = extracted_label_activations[:, 0] - 1
        label_positions = np.array([sentence.labels.index(1) for sentence in self.test_sentences])
        # Confirm that activations are from the position of the specified label
        assert (extracted_positions == label_positions).all()

        # Test extraction selection by token
        self.extractor.model.reset()
        token_sentence_activations = [
            self.extractor._extract_sentence(sentence, lambda pos, token, sentence: token == "hog")
            for sentence in self.corpus.values()
        ]
        extracted_token_activations = _merge_sentence_activations(token_sentence_activations)
        # Confirm that only one activation corresponding to "hog" was extracted
        assert extracted_token_activations.shape[0] == 1
        assert extracted_token_activations[:, -1] == 6

        # Test extraction selection based on misc_info
        self.extractor.model.reset()
        misc_sentence_activations = [
            self.extractor._extract_sentence(
                sentence, lambda pos, token, sentence: sentence.misc_info["quality"] == "delicious"
            )
            for sentence in self.corpus.values()
        ]
        extracted_misc_activations = _merge_sentence_activations(misc_sentence_activations)

        # Confirm that only the first sentence was extracted
        assert (extracted_misc_activations == self.all_activations[:len(self.test_sentences[0].sen), :].numpy()).all()

    # dump_pickle isn't defined in base_extractor but is imported via a from ... import ...
    # statement, therefore this patch path
    @patch('rnnalyse.extractors.base_extractor.dump_pickle')
    def test_extract_average_eos_activations(self, dump_pickle_mock):
        self.extractor.model.reset()
        self.extractor.extract_average_eos_activations()
        # Get the incrementally computed activations that were used as an arg to our mock dump_pickle function
        all_avg_eos_activations, _ = dump_pickle_mock.call_args[0]

        # Confirm the the correct average eos activation was calculated
        eos_activations = [activations[-1, :].unsqueeze(0) for activations in self.test_sentence_activations]
        avg_eos_activation = torch.cat(eos_activations, dim=0).mean(dim=0)
        assert (avg_eos_activation == all_avg_eos_activations[0]["hx"]).all()
