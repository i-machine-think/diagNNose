import os
import pickle
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

import diagnnose.typedefs.config as config
from diagnnose.typedefs.activations import (
    ActivationIndex,
    ActivationKey,
    ActivationName,
    ActivationRanges,
    ActivationTensors,
    Range,
)
from diagnnose.utils.pickle import load_pickle


class ActivationReader:
    """ Reads in pickled activations that have been extracted.

    Parameters
    ----------
    activations_dir : str
        Directory containing the extracted activations
    store_multiple_activations : bool, optional
        Set to true to store multiple activation arrays in RAM at once.
        Defaults to False, meaning that only one activation type will be
        stored in the class.

    Attributes
    ----------
    activations_dir : str
    activations : Optional[np.ndarray]
        Numpy array of activations that are currently read into ram.
    _data_len : int
        Number of extracted activations. Accessed by the property
        self.data_len.
    _activation_ranges : Optional[ActivationRanges]
        Dictionary mapping sentence keys to their respective location
        in the .activations array.
    """

    def __init__(
        self, activations_dir: str, store_multiple_activations: bool = False
    ) -> None:

        assert os.path.exists(
            activations_dir
        ), f"Activations dir not found: {activations_dir}"
        self.activations_dir = activations_dir

        self._tensor_dict: ActivationTensors = {}
        self._data_len: int = -1
        self._activation_ranges: Optional[ActivationRanges] = None

        self.activation_name: Optional[ActivationName] = None
        self.store_multiple_activations = store_multiple_activations

    # TODO: improve docs, clearer formatting etc.
    def __getitem__(self, key: ActivationKey) -> Sequence[Tensor]:
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

        [index] |
        [index, {
            indextype?: 'pos' | 'key' | 'all',
            a_name?: (layer, name),
          }
        ]

        With
            - `indextype` either 'pos', 'key' or 'all'. Defaults to 'pos'.
            - `a_name` an activation name (layer, name) tuple.

        If `a_name` is not provided it should have been set
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
        index, indextype = self._parse_key(key)
        assert self.activations is not None, "self.activations should be set first"

        if indextype == "all":
            return [self.activations[index]]
        if indextype in ["key", "pos"]:
            ranges: List[Range] = self._create_range_from_key(indextype, index)
        else:
            raise KeyError(
                "indextype not understood, should be one of all, key, or pos."
            )

        lengths = [x[1] - x[0] for x in ranges]
        sen_indices = torch.cat([torch.arange(*r) for r in ranges]).to(torch.long)
        activations = self.activations[sen_indices]
        split_activations: Sequence[Tensor] = torch.split(activations, lengths)

        return split_activations

    def _parse_key(self, key: ActivationKey) -> Tuple[ActivationIndex, str]:
        # if key is a tuple it also contains a indextype and/or activation name
        if isinstance(key, tuple):
            index = key[0]
            indextype = key[1].get("indextype", "pos")

            if "a_name" in key[1]:
                self.activations = key[1]["a_name"]
        else:
            index = key
            indextype = "pos"

        if isinstance(index, (int, np.integer)):
            index = [index]

        return index, indextype

    def _create_range_from_key(
        self, indextype: str, index: ActivationIndex
    ) -> List[Range]:
        range_dict = self.activation_ranges
        range_list = list(self.activation_ranges.values())
        if indextype == "key":
            all_ranges: Union[ActivationRanges, List[Range]] = range_dict
        else:
            all_ranges = range_list

        if isinstance(index, (list, np.ndarray)):
            ranges = [all_ranges[r] for r in index]

        elif isinstance(index, Tensor):
            ranges = []
            for i in range(index.size(0)):
                idx = int(index[i].item())
                ranges.append(all_ranges[idx])

        elif isinstance(index, slice):
            start = index.start if index.start else 0
            step = index.step or 1

            if indextype == "key":
                stop = index.stop or max(range_dict.keys()) + 1
                ranges = [
                    r
                    for k, r in range_dict.items()
                    if start <= k < stop and (k + start) % step == 0
                ]
            else:
                stop = index.stop or len(all_ranges)
                ranges = [
                    r
                    for i, r in enumerate(range_list)
                    if start <= i < stop and (i + start) % step == 0
                ]

        else:
            raise KeyError("Type of index is incompatible")

        return ranges

    def __len__(self) -> int:
        return self.data_len

    @property
    def data_len(self) -> int:
        """ data_len is defined based on the activation range of the last sentence """
        if self._data_len == -1:
            self._data_len = list(self.activation_ranges.values())[-1][1]
        return self._data_len

    @property
    def activation_ranges(self) -> ActivationRanges:
        if self._activation_ranges is None:
            ranges_dir = os.path.join(self.activations_dir, "ranges.pickle")
            self._activation_ranges = load_pickle(ranges_dir)
        return self._activation_ranges

    @property
    def activations(self) -> Optional[Tensor]:
        if self.activation_name is None:
            return None
        return self._tensor_dict[self.activation_name]

    @activations.setter
    def activations(self, activation_name: ActivationName) -> None:
        self.activation_name = activation_name
        if activation_name not in self._tensor_dict:
            activations = self.read_activations(activation_name)
            if self.store_multiple_activations:
                self._tensor_dict[activation_name] = activations
            else:
                self._tensor_dict = {activation_name: activations}

    @activations.deleter
    def activations(self) -> None:
        del self._tensor_dict

    def read_activations(self, activation_name: ActivationName) -> Tensor:
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
        filename = os.path.join(self.activations_dir, f"{name}_l{l}.pickle")

        hidden_size = None
        activations = None

        n = 0

        # The activations can be stored as a series of pickle dumps, and
        # are therefore loaded until an EOFError is raised.
        with open(filename, "rb") as f:
            while True:
                try:
                    sen_activations = pickle.load(f)

                    # To make hidden size dependent of data only, the activations array
                    # is created only after observing the first batch of activations.
                    if hidden_size is None:
                        hidden_size = sen_activations.shape[1]
                        activations = torch.empty(
                            (self.data_len, hidden_size), dtype=config.DTYPE
                        )

                    i = len(sen_activations)
                    activations[n : n + i] = sen_activations
                    n += i
                except EOFError:
                    break

        assert (
            activations is not None
        ), f"Reading activations {name}_l{l} returned None, check if file exists and is non-empty."

        return activations
