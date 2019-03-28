import pickle
from typing import List, Optional, Tuple, Union

import numpy as np

from rnnalyse.typedefs.activations import ActivationIndex, ActivationKey, ActivationName
from rnnalyse.typedefs.classifiers import DataDict
from rnnalyse.typedefs.extraction import ActivationRanges, Range
from rnnalyse.utils.paths import load_pickle, trim


class ActivationReader:
    """ Reads in pickled activations that have been extracted.

    Parameters
    ----------
    activations_dir : str
        Directory containing the extracted activations
    label_path : str, optional
        Path to pickle file containing the labels. Defaults to
        labels.pickle in activations_dir if no path has been provided.

    Attributes
    ----------
    activations_dir : str
    label_path : str
    activations : Optional[np.ndarray]
        Numpy array of activations that are currently read into ram.
    _labels : Optional[np.ndarray]
        Numpy array containing the extracted labels. Accessed by the
        property self.labels.
    _data_len : int
        Number of extracted activations. Accessed by the property
        self.data_len.
    _activation_ranges : Optional[ActivationRanges]
        Dictionary mapping sentence keys to their respective location
        in the .activations array.
    """

    def __init__(self,
                 activations_dir: str,
                 label_path: Optional[str] = None) -> None:

        self.activations_dir = trim(activations_dir)

        if label_path is None:
            label_path = f'{self.activations_dir}/labels.pickle'
        self.label_path = label_path

        self._activations: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._data_len: int = -1
        self._activation_ranges: Optional[ActivationRanges] = None

        self.activation_name: Optional[ActivationName] = None
        self.activations: Optional[np.ndarray] = None

    def __getitem__(self, key: ActivationKey) -> np.ndarray:
        """ Provides indexing of activations, indexed by sentence
        position or key, or indexing the activations itself. Indexing
        based on sentence returns all activations belonging to that
        sentence.

        Indexing by position ('pos') refers to the order of extraction,
        selecting the first sentence ([0]) will return all activations
        of the sentence that was extracted first.

        Sentence keys refer to the keys of the labeled corpus that was
        used in the Extractor.

        Indexing can be done by key, index list/np.array, or slicing
        (which is translated into a range of keys).

        The index, indexing type and activation name can optionally
         be provided as the second argument of a tuple as a dictionary.
        This dict is optional, only passing the index is also allowed:

        [index] | ([index], {indextype, concat, a_name})

        With
            - indextype either 'pos', 'key' or 'all', defaults to 'pos'.
            - a_name an activation name (layer, name) tuple.
            - concat a boolean that indicates whether the activations
              should be concatenated or added as a separate tensor dim.

        If activationname is not provided it should have been set
        beforehand like `reader.activations = activationname`.

        Examples:
            reader[8]: activations of the 8th extracted sentence
            reader[8:]: activations of the 8th to final extracted sentence
            reader[8, {'indextype': 'key'}]: activations of a sentence with key 8
            reader[[0,4,6], {'indextype': 'key'}]: activations of sentences with key 0, 4 or 6.
            reader[:20, {'indextype': 'all'}]: the first 20 activations
            reader[8, {'a_name': (0, 'cx')}]: the activations of the cell state in
                the first layer of the 8th extracted sentence.
        """
        index, indextype, concat = self._parse_key(key)
        assert self.activations is not None, 'self.activations should be set first'

        if indextype == 'all':
            return self.activations[index]
        if indextype == 'key':
            ranges = self._create_range_from_key(index)
        else:
            ranges = np.array(list(self.activation_ranges.values()))[index]

        sen_ranges, mask = self._create_indices_from_range(ranges, concat)
        if concat:
            return self.activations[sen_ranges]

        # We allow division by zero here to explicitly set the masked activation values to -inf,
        # so a user would be warned later on if they would try to use such a masked value.
        np.seterr(divide='ignore')
        activations = self.activations[sen_ranges] / np.expand_dims(mask, axis=2)
        np.seterr(divide='warn')

        return activations

    def _parse_key(self, key: ActivationKey) -> Tuple[ActivationIndex, str, bool]:
        indextype = 'pos'
        concat = True
        # if key is a tuple it also contains a indextype and/or activation name
        if isinstance(key, tuple):
            index = key[0]

            indextype = key[1].get('indextype', 'pos')
            concat = key[1].get('concat', True)
            if 'a_name' in key[1]:
                self.activations = key[1]['a_name']
        else:
            index = key

        if isinstance(index, int):
            index = [index]

        return index, indextype, concat

    def _create_range_from_key(self, key: Union[int, slice, List[int], np.ndarray]) -> List[Range]:
        if isinstance(key, (list, np.ndarray)):
            ranges = [self.activation_ranges[r] for r in key]

        elif isinstance(key, slice):
            assert key.step is None or key.step == 1, 'Step slicing not supported for sen key index'
            start = key.start if key.start else 0
            stop = key.stop if key.stop else max(self.activation_ranges.keys()) + 1
            ranges = [r for k, r in self.activation_ranges.items() if start <= k < stop]

        else:
            raise KeyError('Type of index is incompatible')

        return ranges

    @staticmethod
    def _create_indices_from_range(ranges: List[Tuple[int, int]],
                                   concat: bool) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # Concatenate all range indices into a 1d array
        if concat:
            return np.concatenate([range(*r) for r in ranges]), None

        # Create an index array with a separate row for each sentence range
        maxrange = max(x[1]-x[0] for x in ranges)
        indices = np.zeros((len(ranges), maxrange), dtype=int) - 1

        for i, r in enumerate(ranges):
            indices[i, :r[1]-r[0]] = np.array(range(*r))

        mask = indices >= 0

        return indices, mask

    @property
    def labels(self) -> np.ndarray:
        if self._labels is None:
            self._labels = load_pickle(self.label_path)
        return self._labels

    @property
    def data_len(self) -> int:
        """ data_len is defined based on the activation range of the last sentence """
        if self._data_len == -1:
            self._data_len = list(self.activation_ranges.values())[-1][1]
        return self._data_len

    @property
    def activation_ranges(self) -> ActivationRanges:
        if self._activation_ranges is None:
            self._activation_ranges = load_pickle(f'{self.activations_dir}/ranges.pickle')
        return self._activation_ranges

    @property
    def activations(self) -> Optional[np.ndarray]:
        return self._activations

    @activations.setter
    def activations(self, activation_name: Optional[ActivationName]) -> None:
        if activation_name is None:
            self._activations = None
        elif activation_name != self.activation_name:
            self.activation_name = activation_name
            self._activations = self.read_activations(activation_name)

    def read_activations(self, activation_name: ActivationName) -> np.ndarray:
        """ Reads the pickled activations of activation_name

        Parameters
        ----------
        activation_name : ActivationName
            (layer, name) tuple indicating the activations to be read in

        Returns
        -------
        activations : np.ndarray
            Numpy array of activation values
        """
        l, name = activation_name
        filename = f'{name}_l{l}.pickle'

        hidden_size = None
        activations = None

        n = 0

        # The activations can be stored as a series of pickle dumps, and
        # are therefore loaded until an EOFError is raised.
        with open(f'{self.activations_dir}/{filename}', 'rb') as f:
            while True:
                try:
                    sen_activations = pickle.load(f)

                    # To make hidden size dependent of data only, the activations array
                    # is created only after observing the first batch of activations.
                    # TODO: Take care of data_len when using unlabeled corpora! (use np.concatenate)
                    if hidden_size is None:
                        hidden_size = sen_activations.shape[1]
                        activations = np.empty((self.data_len, hidden_size), dtype=np.float32)

                    i = len(sen_activations)
                    activations[n:n+i] = sen_activations
                    n += i
                except EOFError:
                    break

        return activations

    # TODO: move outside this class, to classification specific
    def create_data_split(self,
                          activation_name: ActivationName,
                          data_subset_size: int = -1,
                          train_test_split: float = 0.9) -> DataDict:
        """ Creates train/test data split of activations

        Parameters
        ----------
        activation_name : ActivationName
            (layer, name) tuple indicating the activations to be read in
        data_subset_size : int, optional
            Subset size of data to train on. Defaults to -1, indicating
            the entire data set.
        train_test_split : float
            Percentage of the train/test split. Defaults to 0.9.
        """

        if data_subset_size != -1:
            assert 0 < data_subset_size <= self.data_len, \
                "Size of subset must be positive and not bigger than the whole data set."

        activations = self.read_activations(activation_name)

        data_size = self.data_len if data_subset_size == -1 else data_subset_size
        split = int(data_size * train_test_split)

        indices = np.random.choice(range(data_size), data_size, replace=False)
        train_indices = indices[:split]
        test_indices = indices[split:]

        return {
            'train_x': activations[train_indices],
            'train_y': self.labels[train_indices],
            'test_x': activations[test_indices],
            'test_y': self.labels[test_indices]
        }
