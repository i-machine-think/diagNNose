import os
import pickle
from typing import Optional, Tuple

import torch
from torch import Tensor

import diagnnose.typedefs.config as config
from diagnnose.activations.activation_index import activation_index_to_iterable
from diagnnose.typedefs.activations import (
    ActivationDict,
    ActivationKey,
    ActivationName,
    ActivationRanges,
    SelectionFunc,
)
from diagnnose.utils.pickle import load_pickle


class ActivationReader:
    """ Reads in pickled activations that have been extracted.

    Parameters
    ----------
    activations_dir : str, optional
        Directory containing the extracted activations.
    store_multiple_activations : bool, optional
        Set to true to store multiple activation arrays in RAM at once.
        Defaults to False, meaning that only one activation type will be
        stored in the class.

    Attributes
    ----------
    TODO: update
    """

    def __init__(
        self,
        activations_dir: Optional[str] = None,
        activation_dict: Optional[ActivationDict] = None,
        activation_ranges: Optional[ActivationRanges] = None,
        selection_func: Optional[SelectionFunc] = None,
        store_multiple_activations: bool = False,
    ) -> None:
        if activations_dir is not None:
            assert os.path.exists(
                activations_dir
            ), f"Activations dir not found: {activations_dir}"
        else:
            assert activation_dict is not None
            assert activation_ranges is not None
            assert selection_func is not None

        self.activations_dir = activations_dir
        self.activation_dict: ActivationDict = activation_dict or {}

        self._activation_ranges: Optional[ActivationRanges] = activation_ranges
        self._selection_func: Optional[SelectionFunc] = selection_func

        self.store_multiple_activations = store_multiple_activations

    def __getitem__(self, key: ActivationKey) -> Tuple[Tensor, ...]:
        """
        """
        if isinstance(key, tuple):
            index, activation_name = key
        else:
            assert (
                len(self.activation_dict) == 1
            ), "Activation name must be provided if multiple activations have been extracted"
            index = key
            activation_name = next(iter(self.activation_dict))

        iterable_index = activation_index_to_iterable(
            index, len(self.activation_ranges)
        )
        ranges = [self.activation_ranges[idx] for idx in iterable_index]

        sen_indices = torch.cat([torch.arange(*r) for r in ranges]).to(torch.long)
        activations = self.activations(activation_name)[sen_indices]

        lengths = [x[1] - x[0] for x in ranges]
        split_activations: Tuple[Tensor, ...] = torch.split(activations, lengths)

        return split_activations

    def __len__(self) -> int:
        return self.activation_ranges[-1][1]

    @property
    def activation_ranges(self) -> ActivationRanges:
        if self._activation_ranges is None:
            ranges_path = os.path.join(self.activations_dir, "activation_ranges.pickle")
            self._activation_ranges = load_pickle(ranges_path)
        return self._activation_ranges

    @property
    def selection_func(self) -> SelectionFunc:
        if self._selection_func is None:
            selection_func_path = os.path.join(
                self.activations_dir, "selection_func.dill"
            )
            self._selection_func = load_pickle(selection_func_path, use_dill=True)
        return self._selection_func

    def activations(self, activation_name: ActivationName) -> Tensor:
        activations = self.activation_dict.get(activation_name, None)

        if activations is None:
            activations = self.read_activations(activation_name)
            if not self.store_multiple_activations:
                self.activation_dict = {}
            self.activation_dict[activation_name] = activations

        return activations

    def read_activations(self, activation_name: ActivationName) -> Tensor:
        """ Reads the pickled activations of activation_name

        Parameters
        ----------
        activation_name : ActivationName
            (layer, name) tuple indicating the activations to be read in

        Returns
        -------
        activations : Tensor
            Torch tensor of activation values
        """
        layer, name = activation_name
        filename = os.path.join(self.activations_dir, f"{layer}-{name}.pickle")

        activations = None

        n = 0

        # The activations are stored as a series of pickle dumps, and
        # are therefore loaded until an EOFError is raised.
        with open(filename, "rb") as f:
            while True:
                try:
                    sen_activations = pickle.load(f)

                    # To make hidden size dependent of data only, the activations array
                    # is created only after observing the first batch of activations.
                    if activations is None:
                        hidden_size = sen_activations.shape[1]
                        activations = torch.empty(
                            (len(self), hidden_size), dtype=config.DTYPE
                        )

                    i = len(sen_activations)
                    activations[n : n + i] = sen_activations
                    n += i
                except EOFError:
                    break

        assert activations is not None, (
            f"Reading activations [{layer}, {name}] returned None, "
            f"check if file exists and is non-empty."
        )

        return activations
