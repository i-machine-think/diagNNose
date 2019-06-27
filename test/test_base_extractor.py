"""
Test the code in rnnalayse.extractors.base_extractor.py.
"""

import itertools
import os
import unittest
from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from overrides import overrides
from torch import Tensor

from diagnnose.corpora.create_labels import create_labels_from_corpus
from diagnnose.corpora.import_corpus import import_corpus_from_path
from diagnnose.extractors.base_extractor import Extractor
from diagnnose.models.language_model import LanguageModel
from diagnnose.typedefs.activations import FullActivationDict, PartialActivationDict
from diagnnose.typedefs.corpus import CorpusSentence

from .test_utils import create_sentence_dummy_activations, suppress_print

# GLOBALS
ACTIVATION_DIM = 10
ACTIVATION_NAMES = [(0, "hx"), (0, "cx")]
ACTIVATIONS_DIR = "test/test_data"


class MockLanguageModel(LanguageModel):
    """
    Create a Mock version of the LanguageModel class which returns pre-defined dummy activations.
    """
    def __init__(self,
                 num_layers: int,
                 hidden_size: int,
                 all_tokens: List[str],
                 all_activations: Tensor) -> None:
        super().__init__()
        self.all_tokens = all_tokens
        self.all_activations = all_activations
        self.all_pairs = None

        self.sizes = {
            l: {
                'h': hidden_size, 'c': hidden_size, 'x': hidden_size
            } for l in range(num_layers)
        }
        self.split_order = ['f', 'i', 'g', 'o']
        self.array_type = 'torch'

        self.reset()

    @overrides
    def forward(self,
                token: str,
                _activations: FullActivationDict,
                compute_out: bool = False) -> Tuple[None, FullActivationDict]:
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
    def setUpClass(cls) -> None:
        # Create directory if necessary
        if not os.path.exists(ACTIVATIONS_DIR):
            os.makedirs(ACTIVATIONS_DIR)

        test_corpus = '''The ripe taste improves .\t0 0 1 0 0\tdelicious
        The hog crawled .\t0 1 0 0\thairy
        Move the vat .\t0 0 1 0\tok'''

        corpus_path = os.path.join(ACTIVATIONS_DIR, 'corpus.txt')
        with open(corpus_path, 'w') as f:
            f.write(test_corpus)

        cls.corpus = import_corpus_from_path(corpus_path,
                                             corpus_header=['sen', 'labels', 'quality'])

        # Mock the activations the model produces
        cls.all_tokens = list(itertools.chain(*[item.sen for item in cls.corpus.values()]))
        cls.all_labels = cls._merge_labels([sentence.labels for sentence in cls.corpus.values()])

        cls.test_sentence_activations = []
        identifier_value = 0
        for sentence in cls.corpus.values():
            cls.test_sentence_activations.append(
                create_sentence_dummy_activations(
                    len(sentence.sen), ACTIVATION_DIM, identifier_value
                )
            )
            identifier_value += len(sentence.sen)

        cls.all_activations = torch.cat(cls.test_sentence_activations)

        # Prepare Mock Model
        cls.model = MockLanguageModel(
            num_layers=1, hidden_size=ACTIVATION_DIM, all_tokens=cls.all_tokens,
            all_activations=cls.all_activations
        )

        # Init extractor
        cls.extractor = Extractor(cls.model, cls.corpus, activations_dir=ACTIVATIONS_DIR)
        cls.extractor.activation_names = ACTIVATION_NAMES

    @classmethod
    def tearDownClass(cls) -> None:
        # Delete activations after tests
        if os.listdir(ACTIVATIONS_DIR):
            os.remove(f"{ACTIVATIONS_DIR}/hx_l0.pickle")
            os.remove(f"{ACTIVATIONS_DIR}/cx_l0.pickle")
            os.remove(f"{ACTIVATIONS_DIR}/ranges.pickle")
            os.remove(f"{ACTIVATIONS_DIR}/corpus.txt")

    def test_extract_sentence(self) -> None:
        """ Test _extract_sentence for extracting the activations of whole sentences. """

        # Test extraction of all activations
        self.model.reset()
        sentences_activations, _ = zip(*[
            self.extractor._extract_sentence(sentence)
            for sentence in self.corpus.values()
        ])
        extracted_activations = self._merge_sentence_activations(sentences_activations)
        extracted_labels = create_labels_from_corpus(self.corpus)

        self.assertTrue(
            (extracted_activations == self.all_activations.numpy()).all(),
            "Selection function didn't extract all activations"
        )
        self.assertTrue(
            (extracted_labels == self.all_labels).all(),
            "Selection function didn't extract all labels"
        )

    def test_activation_extraction_by_pos(self) -> None:
        """ Test the _extract_sentence function for extracting activations based on position. """

        self.extractor.model.reset()

        def selection_func(pos: int, _token: str, _sentence: CorpusSentence) -> bool:
            return pos == 2

        pos_sentences_activations, _ = zip(*[
            self.extractor._extract_sentence(sentence, selection_func=selection_func)
            for sentence in self.corpus.values()
        ])

        extracted_pos_activations = self._merge_sentence_activations(pos_sentences_activations)
        extracted_labels = create_labels_from_corpus(self.corpus, selection_func=selection_func)

        # Confirm that only one activation per sentence was extracted
        self.assertEqual(
            extracted_pos_activations.shape[0], 3,
            "More than one sentence was extracted based on position"
        )

        # Confirm that all extracted activations come from position 2
        # Due to the way the dummy activations are created, all their values (except for their
        # unique id value) will be 3
        self.assertTrue(
            (extracted_pos_activations[:, 0] == 3).all(),
            "Sentence was extracted from the wrong position"
        )

        # Confirm the right labels were extracted
        self.assertTrue(
            (extracted_labels == self.all_labels[[2, 7, 11]]).all(),
            "Wrong labels were extracted based on position."
        )

    def test_activation_extraction_by_label(self) -> None:
        """ Test the _extract_sentence function for extracting the activations based on label. """

        self.extractor.model.reset()

        def selection_func(pos: int, _token: str, sentence: CorpusSentence) -> bool:
            return sentence.labels is not None and sentence.labels[pos] == 1

        label_sentence_activations, _ = zip(*[
            self.extractor._extract_sentence(
                sentence, selection_func=selection_func
            )
            for sentence in self.corpus.values()
        ])
        extracted_label_activations = self._merge_sentence_activations(label_sentence_activations)

        # Confirm that only one activation per sentence was extracted
        self.assertEqual(
            extracted_label_activations.shape[0], 3,
            "More than one sentence was extracted based on label"
        )
        extracted_positions = extracted_label_activations[:, 0] - 1
        label_positions = np.array([sentence.labels.index(1) for sentence in self.corpus.values()])
        # Confirm that activations are from the position of the specified label
        self.assertTrue(
            (extracted_positions == label_positions).all(),
            "Wrong activations extracted based on label"
        )

    def test_activation_extraction_by_token(self) -> None:
        """ Test the _extract_sentence function for extracting the activations based on token. """

        self.extractor.model.reset()

        def selection_func(_pos: int, token: str, _sentence: CorpusSentence) -> bool:
            return token == 'hog'

        token_sentence_activations, _ = zip(*[
            self.extractor._extract_sentence(
                sentence, selection_func=selection_func
            )
            for sentence in self.corpus.values()
        ])
        extracted_token_activations = self._merge_sentence_activations(token_sentence_activations)
        extracted_labels = create_labels_from_corpus(self.corpus, selection_func=selection_func)

        # Confirm that only one activation corresponding to "hog" was extracted
        self.assertEqual(
            extracted_token_activations.shape[0], 1,
            "More than one activation extracted by token"
        )
        assert extracted_token_activations[:, -1] == 6

        # Confirm the right labels were extracted
        self.assertTrue(
            (extracted_labels == self.all_labels[[6]]).all(),
            "Wrong labels were extracted based on token."
        )

    def test_activation_extraction_by_misc_info(self) -> None:
        """ Test the _extract_sentence function for extracting activations based on misc info. """

        self.extractor.model.reset()

        def selection_func(_pos: int, _token: str, sentence: CorpusSentence) -> bool:
            return bool(sentence.misc_info['quality'] == 'delicious')

        misc_sentence_activations, _ = zip(*[
            self.extractor._extract_sentence(
                sentence, selection_func=selection_func
            )
            for sentence in self.corpus.values()
        ])
        extracted_misc_activations = self._merge_sentence_activations(misc_sentence_activations)
        extracted_labels = create_labels_from_corpus(self.corpus, selection_func=selection_func)

        # Confirm that only the first sentence was extracted
        expected_activations = self.all_activations[:len(self.corpus[0].sen), :].numpy()
        self.assertTrue(
            (extracted_misc_activations == expected_activations).all(),
            "Wrong sentence extracted based on misc info."
        )

        # Confirm the right labels were extracted
        self.assertTrue(
            (extracted_labels == self.all_labels[:len(self.corpus[0].sen)]).all(),
            "Wrong labels extracted based on misc info."
        )

    @suppress_print
    @patch('diagnnose.activations.activation_writer.ActivationWriter.dump_activations')
    def test_extraction_dumping_args(self, dump_activations_mock: MagicMock) -> None:
        """
        Test whether functions used to dump pickle files during activation extraction are called
        with the right arguments.
        """

        self.extractor.model.reset()
        self.extractor.extract(ACTIVATION_NAMES)
        call_arg = dump_activations_mock.call_args[0][0]

        # Validate function calls
        self.assertEqual(
            dump_activations_mock.call_count, 3,
            "Function was called the wrong number of times."
        )
        self.assertTrue(
            self.is_partial_activation_dict(call_arg),
            "Function was called with wrong type of variable, expected PartialActivationDict."
        )

    @staticmethod
    def _merge_sentence_activations(sentences_activations: List[PartialActivationDict]
                                    ) -> np.ndarray:
        """ Merge activations from different sentences into one single numpy array. """
        return np.array(list(itertools.chain(
            *[sentence_activations[(0, "hx")] for sentence_activations in sentences_activations])
        ))

    @staticmethod
    def _merge_labels(sentence_labels: List[np.array]) -> np.ndarray:
        """ Merge labels from different sentences into a single numpy array. """
        return np.array(list(itertools.chain(*sentence_labels)))

    @staticmethod
    def is_full_activation_dict(var: Any) -> bool:
        """ Check whether a variable is of type FullActivationDict. """

        # This way of checking the type is rather awkward, but there seems to be no function
        # to compare a variable against a subscripted generic - believe me, I also hate this
        first_outer_key = list(var.keys())[0]
        first_value = list(var.values())[0]
        first_inner_key = list(var[first_outer_key].keys())[0]

        return all([
            isinstance(var, dict),
            isinstance(first_outer_key, int),
            isinstance(first_value, dict),
            isinstance(var[first_outer_key[first_inner_key]], Tensor),
        ])

    @staticmethod
    def is_partial_activation_dict(var: Any) -> bool:
        """ Check whether a variable is of type PartialActivationDict. """
        first_key = list(var.keys())[0]
        first_value = list(var.values())[0]

        return all([
            isinstance(var, dict),
            isinstance(first_key, tuple),
            isinstance(first_key[0], int),
            isinstance(first_key[1], str),
            isinstance(first_value, np.ndarray)
        ])
