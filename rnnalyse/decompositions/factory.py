from typing import List, Optional, Tuple, Type, Union

import numpy as np
from sklearn.externals import joblib

from rnnalyse.activations.activation_reader import ActivationReader
from rnnalyse.activations.init_states import InitStates
from rnnalyse.decompositions.cell_decomposer import CellDecomposer
from rnnalyse.decompositions.contextual_decomposer import ContextualDecomposer
from rnnalyse.typedefs.activations import (
    ActivationKey, ActivationName, ActivationNames,
    FullActivationDict, PartialArrayDict)
from rnnalyse.typedefs.classifiers import LinearDecoder
from rnnalyse.utils.paths import trim, camel2snake

from .base_decomposer import BaseDecomposer


class DecomposerFactory:
    """ Creates a BaseDecomposer class for activation decomposition

    Parameters
    ----------
    activations_dir : str
        Path to folder containing extracted activations
    decoder : Union[str, LinearDecoder]
        Path to a pickled decoder or a (w,b) decoder tuple
    num_layers : int
        Number of layers in the language model
    hidden_size : int
        Number of hidden units in the language model
    init_lstm_states_path : str, optional
        Defaults to zero-embeddings, otherwise loads in pickled initial
        cell states.
    """

    def __init__(self,
                 decomposer: Union[Type[BaseDecomposer], str],
                 activations_dir: str,
                 decoder: Union[str, LinearDecoder],
                 num_layers: int,
                 hidden_size: int,
                 init_lstm_states_path: Optional[str] = None) -> None:

        self.decomposer_constructor = self._read_decomposer(decomposer)
        self.activation_reader = ActivationReader(activations_dir, store_multiple_activations=True)

        self.decoder_w, self.decoder_b = self._read_decoder(decoder)

        self.hidden_size = hidden_size
        self.init_cell_state: FullActivationDict = \
            InitStates(num_layers, hidden_size, init_lstm_states_path).create()

    def create(self,
               layer: int,
               sen_index: ActivationKey,
               subsen_index: slice = slice(None, None, None),
               classes: Union[slice, List[int]] = slice(None, None, None)) -> BaseDecomposer:

        decoder = self.decoder_w[classes], self.decoder_b[classes]

        if issubclass(self.decomposer_constructor, CellDecomposer):
            activation_names = [(layer, name) for name in ['f_g', 'o_g', 'hx', 'cx', 'icx', '0cx']]
        elif issubclass(self.decomposer_constructor, ContextualDecomposer):
            activation_names = [(l, 'icx') for l in range(layer+1)] \
                               + [(l, '0cx') for l in range(layer+1)] \
                               + [(0, 'emb')]
        else:
            raise ValueError('Decomposer constructor not understood')

        activation_dict, final_index = self._create_activations(activation_names, sen_index,
                                                                subsen_index)

        decomposer = self.decomposer_constructor(decoder, activation_dict, final_index, layer)

        return decomposer

    def _read_activations(self,
                          a_name: ActivationName,
                          sen_index: ActivationKey) -> np.ma.MaskedArray:
        activation_key_config = {'indextype': 'key', 'concat': False, 'a_name': a_name}
        return self.activation_reader[sen_index, activation_key_config]

    def _create_activations(self,
                            activation_names: ActivationNames,
                            sen_index: ActivationKey,
                            subsen_index: slice = slice(None, None, None),
                            ) -> Tuple[PartialArrayDict, np.ndarray]:
        activation_dict: PartialArrayDict = {}

        for a_name in activation_names:
            if a_name[1] in ['icx', '0cx', 'ihx', '0hx']:
                activation = self._create_init_activations(a_name, sen_index, subsen_index)
            else:
                activation = self._read_activations(a_name, sen_index)
                activation = activation[:, subsen_index]

            activation_dict[a_name] = activation

        final_index = self._create_final_index_array(sen_index, subsen_index)

        return activation_dict, final_index

    def _create_init_activations(self,
                                 activation_name: ActivationName,
                                 sen_index: ActivationKey,
                                 subsen_index: slice) -> np.ndarray:
        """ Creates the init state activations. """

        # name can only be icx/ihx or 0cx/0hx
        layer, name = activation_name
        activations = self._read_activations((layer, name[1:]), sen_index)

        batch_size = activations.shape[0]

        if name[0] == '0':
            return np.zeros((batch_size, 1, self.hidden_size))

        if subsen_index.start == 0 or subsen_index.start is None:
            init_state = self.init_cell_state[layer][name[1:]].numpy()
            init_state = np.tile(init_state, (batch_size, 1))
        else:
            init_state = activations[:, subsen_index.start - 1]

        return np.expand_dims(init_state, 1)

    def _create_final_index_array(self,
                                  sen_index: ActivationKey,
                                  subsen_index: slice) -> np.ndarray:
        """ Computes the final index of each sentence in the batch. """

        cell_states = self._read_activations((0, 'cx'), sen_index)

        batch_size = cell_states.shape[0]

        final_index = np.sum(np.all(1 - cell_states.mask, axis=2), axis=1) - 1
        if subsen_index.stop:
            assert np.all(final_index < subsen_index.stop), \
                'Subsentence index can\'t be longer than sentence itself'
            final_index = np.array([subsen_index.stop - 1] * batch_size)

        start = subsen_index.start if subsen_index.start else 0
        assert start >= 0, 'Subsentence index can\'t be negative'
        final_index -= start

        return final_index

    @staticmethod
    def _read_decomposer(decomposer_constructor: Union[str, Type[BaseDecomposer]]
                         ) -> Type[BaseDecomposer]:
        if isinstance(decomposer_constructor, str):
            # Import Decomposer class from string, assumes module name to be snake case variant
            # of CamelCased Decomposer class. Taken from: https://stackoverflow.com/a/30941292
            from importlib import import_module
            module_name = camel2snake(decomposer_constructor)
            module = import_module(f'rnnalyse.decompositions.{module_name}')
            decomposer: Type[BaseDecomposer] = getattr(module, decomposer_constructor)
            return decomposer

        return decomposer_constructor

    @staticmethod
    def _read_decoder(decoder: Union[str, LinearDecoder]) -> LinearDecoder:
        if isinstance(decoder, str):
            classifier = joblib.load(trim(decoder))
            decoder_w = classifier.coef_
            decoder_b = classifier.intercept_
        else:
            decoder_w, decoder_b = decoder

        return decoder_w, decoder_b
