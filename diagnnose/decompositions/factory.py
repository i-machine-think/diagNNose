from importlib import import_module
from typing import List, Optional, Tuple, Type, Union

import numpy as np
from sklearn.externals import joblib

from diagnnose.activations.activation_reader import ActivationReader
from diagnnose.activations.init_states import InitStates
from diagnnose.decompositions import CellDecomposer, ContextualDecomposer
from diagnnose.models.import_model import import_decoder_from_model
from diagnnose.models.language_model import LanguageModel
from diagnnose.typedefs.activations import (
    ActivationKey,
    ActivationName,
    ActivationNames,
    FullActivationDict,
    PartialArrayDict,
)
from diagnnose.typedefs.classifiers import LinearDecoder
from diagnnose.utils.paths import camel2snake

from .base_decomposer import BaseDecomposer


class DecomposerFactory:
    """ Creates a BaseDecomposer class for activation decomposition

    Parameters
    ----------
    model : LanguageModel
        LanguageModel for which decomposition will be performed
    decomposer : str
        String of the decomposition class name, either CellDecomposer or
        ContextualDecomposer
    activations_dir : str
        Path to folder containing extracted activations
    decoder : Union[str, LinearDecoder]
        Path to a pickled decoder or a (w,b) decoder tuple
    init_lstm_states_path : str, optional
        Defaults to zero-embeddings, otherwise loads in pickled initial
        cell states.
    """

    def __init__(
        self,
        model: LanguageModel,
        decomposer: str,
        activations_dir: str,
        decoder: Optional[str] = None,
        init_lstm_states_path: Optional[str] = None,
    ) -> None:

        # Import Decomposer class from string, assumes module name to be snake case variant
        # of CamelCased Decomposer class. Taken from: https://stackoverflow.com/a/30941292
        module_name = camel2snake(decomposer)
        module = import_module(f"diagnnose.decompositions.{module_name}")
        self.decomposer_constructor: Type[BaseDecomposer] = getattr(module, decomposer)

        self.activation_reader = ActivationReader(
            activations_dir, store_multiple_activations=True
        )
        self.model = model

        self.decoder_w, self.decoder_b = self._read_decoder(decoder)

        self.init_cell_state: FullActivationDict = InitStates(
            model, init_lstm_states_path
        ).create()

    def create(
        self,
        sen_index: ActivationKey,
        subsen_index: slice = slice(None, None, None),
        classes: Union[slice, List[int]] = slice(None, None, None),
    ) -> BaseDecomposer:
        """ Creates an instance of a BaseDecomposer.

        Parameters
        ----------
        sen_index : ActivationKey
            Denotes the sentence index or range of indices for which the
            decomposition should be performed.
        subsen_index : slice, optional
            Denotes slice of sentences on which decomposition should be
            performed, allowing only a subsentence to be taken into
            account.
        classes : slice | List[int], optional
            Denotes the the class indices of the model decoder for which
            the decomposed scores should be calculated. Defaults to the
            entire vocabulary.

        Returns
        -------
        decomposer : BaseDecomposer
            BaseDecomposer instance pertaining to the provided
            parameters.
        """

        decoder = self.decoder_w[classes], self.decoder_b[classes]

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
            activation_names, sen_index, subsen_index
        )

        decomposer = self.decomposer_constructor(
            self.model, activation_dict, decoder, final_index
        )

        return decomposer

    def _read_activations(
        self, a_name: ActivationName, sen_index: ActivationKey
    ) -> np.ma.MaskedArray:
        activation_key_config = {"indextype": "key", "concat": False, "a_name": a_name}
        return self.activation_reader[sen_index, activation_key_config]

    def _create_activations(
        self,
        activation_names: ActivationNames,
        sen_index: ActivationKey,
        subsen_index: slice = slice(None, None, None),
    ) -> Tuple[PartialArrayDict, np.ndarray]:
        activation_dict: PartialArrayDict = {}

        for a_name in activation_names:
            if a_name[1] in ["icx", "0cx", "ihx", "0hx"]:
                activation = self._create_init_activations(
                    a_name, sen_index, subsen_index
                )
            else:
                activation = self._read_activations(a_name, sen_index)
                activation = activation[:, subsen_index]

            activation_dict[a_name] = activation

        final_index = self._create_final_index_array(sen_index, subsen_index)

        return activation_dict, final_index

    def _create_init_activations(
        self,
        activation_name: ActivationName,
        sen_index: ActivationKey,
        subsen_index: slice,
    ) -> np.ndarray:
        """ Creates the init state activations. """

        # name can only be icx/ihx or 0cx/0hx
        layer, name = activation_name
        activations = self._read_activations((layer, name[1:]), sen_index)

        batch_size = activations.shape[0]

        if name[0] == "0":
            cell_type = name[1]
            return np.zeros((batch_size, 1, self.model.sizes[layer][cell_type]))

        if subsen_index.start == 0 or subsen_index.start is None:
            init_state = self.init_cell_state[layer][name[1:]]
            if self.model.array_type == "torch":
                init_state = init_state.numpy()
            init_state = np.tile(init_state, (batch_size, 1))
        else:
            init_state = activations[:, subsen_index.start - 1]

        return np.expand_dims(init_state, 1)

    def _create_final_index_array(
        self, sen_index: ActivationKey, subsen_index: slice
    ) -> np.ndarray:
        """ Computes the final index of each sentence in the batch. """

        cell_states = self._read_activations((0, "cx"), sen_index)

        batch_size = cell_states.shape[0]

        final_index = np.sum(np.all(1 - cell_states.mask, axis=2), axis=1) - 1
        if subsen_index.stop:
            assert np.all(
                final_index >= subsen_index.stop - 1
            ), "Subsentence index can't be longer than sentence itself"
            final_index = np.array([subsen_index.stop - 1] * batch_size)

        start = subsen_index.start if subsen_index.start else 0
        assert start >= 0, "Subsentence index can't be negative"
        final_index -= start

        return final_index

    def _read_decoder(self, decoder_path: Optional[str]) -> LinearDecoder:
        if decoder_path:
            classifier = joblib.load(decoder_path)
            decoder_w = classifier.coef_
            decoder_b = classifier.intercept_
        else:
            decoder_w, decoder_b = import_decoder_from_model(self.model)

        return decoder_w, decoder_b
