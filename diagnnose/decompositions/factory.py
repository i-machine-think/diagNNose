from importlib import import_module
from typing import List, Optional, Tuple, Type

import torch
from numpy import ndarray
from sklearn.externals import joblib
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

import diagnnose.typedefs.config as config
from diagnnose.activations.activation_index import (
    activation_index_len,
    activation_index_to_iterable,
)
from diagnnose.activations.activation_reader import ActivationReader
from diagnnose.decompositions import CellDecomposer, ContextualDecomposer
from diagnnose.extractors.base_extractor import Extractor
from diagnnose.models.import_model import import_decoder_from_model
from diagnnose.models.lm import LanguageModel
from diagnnose.typedefs.activations import (
    ActivationIndex,
    ActivationName,
    ActivationNames,
    ActivationTensors,
)
from diagnnose.typedefs.classifiers import LinearDecoder
from diagnnose.typedefs.corpus import Corpus
from diagnnose.utils.misc import suppress_print

from .base_decomposer import BaseDecomposer


class DecomposerFactory:
    """ Creates a BaseDecomposer class for activation decomposition

    Parameters
    ----------
    model : LanguageModel
        LanguageModel for which decomposition will be performed
    activations_dir : str
        Path to folder containing extracted activations
    create_new_activations : bool, optional
        Toggle to create new extracted activations, that will be saved
        to `activations_dir`. `corpus` should be provided if set to
        True. Defaults to False.
    corpus : Corpus, optional
        If the activations have not been extracted yet a Corpus object
        must be passed for which the decomposer will be created.
        The activations will be temporarily extracted in that case.
    sen_ids : ActivationIndex, optional
        ActivationIndex of the corpus items for which the Factory should
        be created. If not provided the full corpus will be used.
    decomposer : str
        String of the decomposition class name, either `CellDecomposer`
        or `ContextualDecomposer`. Defaults to `ContextualDecomposer`.
    """

    def __init__(
        self,
        model: LanguageModel,
        activations_dir: str,
        create_new_activations: bool = False,
        corpus: Optional[Corpus] = None,
        sen_ids: ActivationIndex = slice(None, None),
        decomposer: str = "ContextualDecomposer",
    ) -> None:
        self.model = model

        module = import_module(f"diagnnose.decompositions")
        self.decomposer_constructor: Type[BaseDecomposer] = getattr(module, decomposer)

        if create_new_activations:
            assert (
                corpus is not None
            ), "Corpus should be provided if no activations_dir is passed."
            self._extract_activations(activations_dir, sen_ids, corpus)
        self.activation_reader = ActivationReader(
            activations_dir, store_multiple_activations=True
        )

    def create(
        self,
        sen_ids: ActivationIndex,
        subsen_index: slice = slice(None, None, None),
        classes: Optional[ActivationIndex] = None,
        extra_classes: Optional[List[int]] = None,
    ) -> BaseDecomposer:
        """ Creates an instance of a BaseDecomposer.

        Parameters
        ----------
        sen_ids : ActivationIndex
            Denotes the sentence indices for which the decomposition
            should be performed.
        subsen_index : slice, optional
            Denotes slice of sentences on which decomposition should be
            performed, allowing only a subsentence to be taken into
            account.
        classes : slice | List[int], optional
            Denotes the the class indices of the model decoder for which
            the decomposed scores should be calculated. Defaults to
            None, indicating no class predictions will be created. In
            that case the decomposed states themselves are returned.
            Pass `slice(None, None, None)` to create predictions for
            the full model.
        extra_classes : List[int], optional
            List of indices where optional extra classes can be
            calculated. This makes it easier to calculate multiple
            output classes at the same sentence position. If provided
            the final decomposed states will be replaced by the states
            at the provided indices.

        Returns
        -------
        decomposer : BaseDecomposer
            BaseDecomposer instance pertaining to the provided
            parameters.
        """
        batch_size = activation_index_len(sen_ids)
        decoder = self._read_decoder(classes, batch_size)

        if issubclass(self.decomposer_constructor, CellDecomposer):
            activation_names = [
                (self.model.num_layers - 1, name)
                for name in ["f_g", "o_g", "hx", "cx", "icx", "0cx"]
            ]
        elif issubclass(self.decomposer_constructor, ContextualDecomposer):
            activation_names = [(0, "emb")]
            for l in range(self.model.num_layers):
                activation_names.extend([(l, "cx"), (l, "hx")])
                activation_names.extend([(l, "icx"), (l, "ihx")])
        else:
            raise ValueError("Decomposer constructor not understood")

        activation_dict, final_index = self._create_activations(
            activation_names, sen_ids, subsen_index
        )

        if extra_classes is None:
            extra_classes = []

        decomposer = self.decomposer_constructor(
            self.model, activation_dict, decoder, final_index, extra_classes
        )

        return decomposer

    def _create_activations(
        self,
        activation_names: ActivationNames,
        sen_ids: ActivationIndex,
        subsen_index: slice = slice(None, None, None),
    ) -> Tuple[ActivationTensors, Tensor]:
        activation_dict: ActivationTensors = {}

        final_index = self._create_final_index(sen_ids, subsen_index)
        batch_size = final_index.size(0)

        for a_name in activation_names:
            if a_name[1] in ["icx", "0cx", "ihx", "0hx"]:
                activations = self._create_init_activations(
                    a_name, sen_ids, subsen_index, batch_size
                )
            else:
                activations = self._read_activations(a_name, sen_ids)
                activations = activations[:, subsen_index]

            activation_dict[a_name] = activations

        return activation_dict, final_index

    def _create_init_activations(
        self,
        activation_name: ActivationName,
        sen_ids: ActivationIndex,
        subsen_index: slice,
        batch_size: int,
    ) -> Tensor:
        """ Creates the init state activations. """

        # name can only be icx/ihx or 0cx/0hx
        layer, name = activation_name

        if name[0] == "0":
            cell_type = name[1]
            return torch.zeros(
                (batch_size, self.model.sizes[layer][cell_type]), dtype=config.DTYPE
            )

        if subsen_index.start == 0 or subsen_index.start is None:
            init_state: Tensor = self.model.init_hidden(batch_size)[layer, name[1:]]
        else:
            activations = self._read_activations((layer, name[1:]), sen_ids)
            init_state = activations[:, subsen_index.start - 1]

        # Shape: (batch_size, nhid)
        return init_state

    def _create_final_index(
        self, sen_ids: ActivationIndex, subsen_index: slice
    ) -> Tensor:
        """ Computes the final index of each sentence in the batch. """
        final_indices = self._calc_batch_lens(sen_ids) - 1
        batch_size = final_indices.size(0)

        if subsen_index.stop:
            assert (
                torch.min(final_indices) >= subsen_index.stop - 1
            ), "Subsentence index can't be longer than sentence itself"
            if subsen_index.stop < 0:
                final_indices += subsen_index.stop
            else:
                final_indices = torch.tensor([subsen_index.stop - 1] * batch_size)

        start = subsen_index.start if subsen_index.start else 0
        assert start >= 0, "Subsentence index can't be negative"
        final_indices -= start

        return final_indices.to(torch.long)

    def _calc_batch_lens(self, sen_ids: ActivationIndex) -> Tensor:
        """ Returns final indices of current batch.

        Parameters
        ----------
        sen_ids : ActivationIndex
            Indices of the current batch.

        Returns
        -------
        batch_lens : Tensor
            Tensor of the sentence length of each batch item.
        """
        activation_key_config = {"indextype": "pos", "a_name": (0, "cx")}
        cell_states = self.activation_reader[sen_ids, activation_key_config]
        batch_lens = torch.tensor([cx.size(0) for cx in cell_states])

        return batch_lens

    def _read_activations(
        self, a_name: ActivationName, sen_ids: ActivationIndex
    ) -> Tensor:
        activation_key_config = {"indextype": "pos", "a_name": a_name}
        activations = self.activation_reader[sen_ids, activation_key_config]
        padded_activations: Tensor = pad_sequence(
            list(activations), batch_first=True, padding_value=float("nan")
        )
        return padded_activations

    def _read_decoder(
        self,
        classes: Optional[ActivationIndex],
        batch_size: int,
        decoder_path: Optional[str] = None,
    ) -> LinearDecoder:
        if decoder_path is not None:
            classifier = joblib.load(decoder_path)
            decoder_w = classifier.coef_
            decoder_b = classifier.intercept_
        else:
            decoder_w, decoder_b = import_decoder_from_model(self.model)

        if isinstance(classes, int):
            classes = torch.tensor([classes])
        elif isinstance(classes, list):
            classes = torch.tensor(classes)
        elif isinstance(classes, ndarray):
            classes = torch.from_numpy(classes)
        elif isinstance(classes, slice):
            classes = torch.tensor(activation_index_to_iterable(classes))
        elif classes is None:
            classes = torch.tensor([]).to(torch.long)

        if len(classes.shape) == 1:
            classes = classes.repeat(batch_size, 1)
        else:
            assert classes.size(0) == batch_size, (
                f"First dimension of classes not equal to batch_size:"
                f" {classes.size(0)} != {batch_size} (bsz)"
            )

        decoder_w = decoder_w[classes].permute(0, 2, 1)
        decoder_b = decoder_b[classes]

        return decoder_w, decoder_b

    def _extract_activations(
        self, activations_dir: str, sen_ids: ActivationIndex, corpus: Corpus
    ) -> None:
        activation_names = self._get_activation_names()

        if sen_ids.stop is None:
            sen_ids = slice(sen_ids.start, len(corpus), sen_ids.step)

        all_examples = list(corpus.examples)  # create copy of full corpus
        corpus.examples = [
            corpus.examples[idx] for idx in activation_index_to_iterable(sen_ids)
        ]  # discard all other items
        extractor = Extractor(self.model, corpus, activations_dir, activation_names)

        self._extract(extractor)
        corpus.examples = all_examples  # restore initial corpus

    def _get_activation_names(self) -> ActivationNames:
        activation_names: ActivationNames = []

        if issubclass(self.decomposer_constructor, ContextualDecomposer):
            for l in range(self.model.num_layers):
                activation_names.extend([(l, "cx"), (l, "hx")])
            activation_names.append((0, "emb"))
        else:
            activation_names.extend(
                [
                    (self.model.num_layers - 1, name)
                    for name in ["f_g", "o_g", "hx", "cx", "icx", "0cx"]
                ]
            )

        return activation_names

    @staticmethod
    @suppress_print
    def _extract(extractor: Extractor) -> None:
        extractor.extract(batch_size=1024, dynamic_dumping=False)
