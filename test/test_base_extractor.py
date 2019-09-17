"""
Test the code in diagnnose.extractors.base_extractor.py.
"""

import itertools
import os
import unittest
from typing import Any, List, Tuple
from unittest.mock import MagicMock, patch

import torch
from overrides import overrides
from torch import Tensor
from torchtext.data import Example

import diagnnose.typedefs.config as config
from diagnnose.corpus import import_corpus
from diagnnose.corpus.create_iterator import create_iterator
from diagnnose.corpus.create_labels import create_labels_from_corpus
from diagnnose.extractors.base_extractor import Extractor
from diagnnose.models.lm import LanguageModel
from diagnnose.typedefs.activations import ActivationTensors, SelectFunc
from diagnnose.utils.misc import suppress_print

from .test_utils import create_sentence_dummy_activations

# GLOBALS
ACTIVATION_DIM = 10
ACTIVATION_NAMES = [(0, "hx"), (0, "cx")]
ACTIVATIONS_DIR = "test/test_data"


class MockLanguageModel(LanguageModel):
    """
    Create a Mock version of the LanguageModel class which returns pre-defined dummy activations.
    """

    def __init__(
        self,
        num_layers: int,
        hidden_size: int,
        all_tokens: List[str],
        all_activations: Tensor,
    ) -> None:
        self.all_tokens = all_tokens
        self.all_activations = all_activations
        self.all_pairs = None

        self.sizes = {
            l: {"h": hidden_size, "c": hidden_size, "x": hidden_size}
            for l in range(num_layers)
        }
        self.split_order = ["f", "i", "g", "o"]
        self.array_type = "torch"

        self.reset()

        super().__init__()

    @overrides
    def forward(
        self,
        token: torch.Tensor,
        _activations: ActivationTensors,
        compute_out: bool = False,
    ) -> Tuple[None, ActivationTensors]:
        # Consume next activation, make sure it's the right token
        next_token, next_activation = next(self.all_pairs)
        assert token.item() == next_token

        if len(next_activation.shape) == 1:
            next_activation = next_activation.unsqueeze(dim=0)

        return None, {(0, "hx"): next_activation, (0, "cx"): next_activation}

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

        test_corpus = """The ripe taste improves .\t0 0 1 0 0\tdelicious
        The hog crawled .\t0 1 0 0\thairy
        Move the vat .\t0 0 1 0\tok"""

        corpus_path = os.path.join(ACTIVATIONS_DIR, "corpus.txt")
        with open(corpus_path, "w") as f:
            f.write(test_corpus)

        cls.corpus = import_corpus(
            corpus_path, header=["sen", "labels", "quality"], vocab_from_corpus=True
        )
        cls.examples = cls.corpus.examples
        cls.iterator = create_iterator(cls.corpus, batch_size=1)

        # Mock the activations the model produces
        cls.all_words = list(itertools.chain(*[item.sen for item in cls.corpus]))
        cls.all_tokens = [cls.corpus.vocab.stoi[w] for w in cls.all_words]
        cls.all_labels = cls._merge_labels([example.labels for example in cls.corpus])

        test_sentence_activations = []
        identifier_value = 0
        for example in cls.corpus:
            test_sentence_activations.append(
                create_sentence_dummy_activations(
                    len(example.sen), ACTIVATION_DIM, identifier_value
                )
            )
            identifier_value += len(example.sen)

        cls.all_activations = torch.cat(test_sentence_activations)

        # Prepare Mock Model
        cls.model = MockLanguageModel(
            num_layers=1,
            hidden_size=ACTIVATION_DIM,
            all_tokens=cls.all_tokens,
            all_activations=cls.all_activations,
        )
        cls.model.set_init_states()

        # Init extractor
        cls.extractor = Extractor(
            cls.model,
            cls.corpus,
            activations_dir=ACTIVATIONS_DIR,
            activation_names=ACTIVATION_NAMES,
        )
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

        def selection_func(_sen_id: int, _pos: int, _example: Example) -> bool:
            return True

        extracted_activations, extracted_labels = self._base_extract(selection_func)

        self.assertTrue(
            (extracted_activations == self.all_activations).all(),
            "Selection function didn't extract all activations",
        )
        self.assertTrue(
            (extracted_labels == self.all_labels).all(),
            "Selection function didn't extract all labels",
        )

    def test_activation_extraction_by_pos(self) -> None:
        """ Test the _extract_sentence function for extracting activations based on position. """

        def selection_func(_sen_id: int, pos: int, _example: Example) -> bool:
            return pos == 2

        extracted_activations, extracted_labels = self._base_extract(selection_func)

        # Confirm that only one activation per sentence was extracted
        self.assertEqual(
            extracted_activations.shape[0],
            3,
            "More than one sentence was extracted based on position",
        )

        # Confirm that all extracted activations come from position 2
        # Due to the way the dummy activations are created, all their values (except for their
        # unique id value) will be 3
        self.assertTrue(
            (extracted_activations[:, 0] == 3).all(),
            "Sentence was extracted from the wrong position",
        )

        # Confirm the right labels were extracted
        self.assertTrue(
            (extracted_labels == self.all_labels[[2, 7, 11]]).all(),
            "Wrong labels were extracted based on position.",
        )

    def test_activation_extraction_by_label(self) -> None:
        """ Test the _extract_sentence function for extracting the activations based on label. """

        def selection_func(_sen_id: int, pos: int, example: Example) -> bool:
            return (
                getattr(example, "labels") is not None
                and getattr(example, "labels")[pos] == 1
            )

        extracted_activations, extracted_labels = self._base_extract(selection_func)

        # Confirm that only one activation per sentence was extracted
        self.assertEqual(
            extracted_activations.shape[0],
            3,
            "More than one sentence was extracted based on label",
        )
        extracted_positions = extracted_activations[:, 0] - 1
        label_positions = torch.tensor(
            [example.labels.index(1) for example in self.examples], dtype=config.DTYPE
        )
        # Confirm that activations are from the position of the specified label
        self.assertTrue(
            (extracted_positions == label_positions).all(),
            "Wrong activations extracted based on label",
        )

    def test_activation_extraction_by_token(self) -> None:
        """ Test the _extract_sentence function for extracting the activations based on token. """

        def selection_func(_sen_id: int, pos: int, example: Example) -> bool:
            return (
                getattr(example, "sen") is not None
                and getattr(example, "sen")[pos] == "hog"
            )

        extracted_activations, extracted_labels = self._base_extract(selection_func)

        # Confirm that only one activation corresponding to "hog" was extracted
        self.assertEqual(
            extracted_activations.shape[0],
            1,
            "More than one activation extracted by token",
        )
        assert extracted_activations[:, -1] == 6

        # Confirm the right labels were extracted
        self.assertTrue(
            (extracted_labels == self.all_labels[[6]]).all(),
            "Wrong labels were extracted based on token.",
        )

    def test_activation_extraction_by_misc_info(self) -> None:
        """ Test the _extract_sentence function for extracting activations based on misc info. """

        def selection_func(_sen_id: int, _pos: int, example: Example) -> bool:
            return hasattr(example, "quality") and bool(
                getattr(example, "quality", "") == "delicious"
            )

        extracted_activations, extracted_labels = self._base_extract(selection_func)

        # Confirm that only the first sentence was extracted
        expected_activations = self.all_activations[: len(self.corpus[0].sen), :]
        self.assertTrue(
            (extracted_activations == expected_activations).all(),
            "Wrong sentence extracted based on misc info.",
        )

        # Confirm the right labels were extracted
        self.assertTrue(
            (extracted_labels == self.all_labels[: len(self.corpus[0].sen)]).all(),
            "Wrong labels extracted based on misc info.",
        )

    @suppress_print
    @patch(
        "diagnnose.activations.activation_writer.ActivationWriter.concat_pickle_dumps"
    )
    @patch("diagnnose.activations.activation_writer.ActivationWriter.dump_activations")
    def test_extraction_dumping_args(
        self, dump_activations_mock: MagicMock, _concat_pickle_dumps_mock: MagicMock
    ) -> None:
        """
        Test whether functions used to dump pickle files during activation extraction are called
        with the right arguments.
        """

        self.extractor.model.reset()
        self.extractor.extract()
        call_arg = dump_activations_mock.call_args[0][0]

        # Validate function calls
        self.assertEqual(
            dump_activations_mock.call_count,
            3,
            "Function was called the wrong number of times.",
        )
        self.assertTrue(
            self.is_tensor_dict(call_arg),
            "Function was called with wrong type of variable, expected PartialActivationDict.",
        )

    def _base_extract(self, selection_func: SelectFunc) -> Tuple[Tensor, Tensor]:
        self.model.reset()

        sen_activations = []
        for i, batch in enumerate(self.iterator):
            sen_activation = self.extractor._extract_sentence(batch, i, selection_func)[
                0
            ]
            for j in sen_activation.keys():
                sen_activations.append(sen_activation[j])

        extracted_activations = self._merge_sentence_activations(sen_activations)
        extracted_labels = create_labels_from_corpus(
            self.corpus, selection_func=selection_func
        )

        return extracted_activations, extracted_labels

    @staticmethod
    def _merge_sentence_activations(
        sentences_activations: List[ActivationTensors]
    ) -> Tensor:
        """ Merge activations from different sentences into one single numpy array. """
        return torch.cat(
            [
                sentence_activations[(0, "hx")]
                for sentence_activations in sentences_activations
            ],
            dim=0,
        )

    @staticmethod
    def _merge_labels(sentence_labels: List[List[int]]) -> Tensor:
        """ Merge labels from different sentences into a single numpy array. """
        return torch.tensor([x for l in sentence_labels for x in l])

    @staticmethod
    def is_tensor_dict(var: Any) -> bool:
        """ Check whether a variable is of type TensorDict. """
        first_key = list(var.keys())[0]
        first_value = list(var.values())[0]

        return all(
            [
                isinstance(var, dict),
                isinstance(first_key, tuple),
                isinstance(first_key[0], int),
                isinstance(first_key[1], str),
                isinstance(first_value, Tensor),
            ]
        )
