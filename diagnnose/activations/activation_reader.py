import os
import pickle
from typing import Iterator, Optional, Union

import torch
from torch import Tensor

from diagnnose.activations.activation_index import activation_index_to_iterable
from diagnnose.typedefs.activations import (
    ActivationDict,
    ActivationKey,
    ActivationName,
    ActivationNames,
    ActivationRanges,
    SelectionFunc,
)
from diagnnose.utils.pickle import load_pickle


class ActivationReader:
    """Reads in pickled activations that have been extracted.

    An ``ActivationReader`` can also be created directly from an
    ``ActivationDict``, in which case the corresponding
    ``ActivationRanges`` and ``SelectionFunc`` should be provided too.

    Parameters
    ----------
    activations_dir : str, optional
        Directory containing the extracted activations.
    activation_dict : ActivationDict, optional
        If activations have not been extracted to disk, the
        activation_dict containing all extracted activations can be
        provided directly as well.
    activation_names : ActivationNames, optional
        Activation names, provided as a list of ``(layer, name)``
        tuples. If not provided the index to
        :func:`~diagnnose.activations.ActivationReader.__getitem__`
        must always contain the activation_name that is being requested,
        as the ``ActivationReader`` can not infer it automatically.
    activation_ranges : ActivationRanges, optional
        ``ActivationRanges`` dictionary that should be provided if
        ``activation_dict`` is passed directly.
    selection_func : SelectionFunc, optional
        ``SelectionFunc`` that was used for extraction and that should
        be passed if ``activation_dict`` is passed directly.
    store_multiple_activations : bool, optional
        Set to true to store multiple activation arrays in RAM at once.
        Defaults to False, meaning that only one activation type will be
        stored in the class.
    cat_activations : bool, optional
        Toggle to concatenate the activations returned by
        :func:`~diagnnose.activations.ActivationReader.__getitem__`.
        Otherwise the activations will be split into a tuple with each
        each tuple item containing the activations of one sentence.
    """

    def __init__(
        self,
        activations_dir: Optional[str] = None,
        activation_dict: Optional[ActivationDict] = None,
        activation_names: Optional[ActivationNames] = None,
        activation_ranges: Optional[ActivationRanges] = None,
        selection_func: Optional[SelectionFunc] = None,
        store_multiple_activations: bool = False,
        cat_activations: bool = False,
    ) -> None:
        if activations_dir is not None:
            assert os.path.exists(
                activations_dir
            ), f"Activations dir not found: {activations_dir}"
            assert (
                activation_dict is None
            ), "activations_dir and activations_dict can not be provided simultaneously"
        else:
            assert activation_dict is not None
            assert activation_ranges is not None
            assert selection_func is not None

        self.activations_dir = activations_dir
        self.activation_dict: ActivationDict = activation_dict or {}
        self.activation_names: ActivationNames = activation_names or list(
            self.activation_dict.keys()
        )

        self._activation_ranges: Optional[ActivationRanges] = activation_ranges
        self._selection_func: Optional[SelectionFunc] = selection_func

        self.store_multiple_activations = store_multiple_activations
        self.cat_activations = cat_activations

    def __getitem__(self, key: ActivationKey) -> Union[Tensor, Iterator[Tensor]]:
        """Allows for concise and efficient indexing of activations.

        The ``key`` argument should be either an ``ActivationIndex``
        (i.e. an iterable that can be used to index a tensor), or a
        ``(index, activation_name)`` tuple. An ``activation_name`` is
        a tuple of shape ``(layer, name)``.

        If multiple activation_names have been extracted the
        ``activation_name`` must be provided, otherwise it can be left
        out.

        The return value is a generator of tensors, with each tensor of
        shape (sen_len, nhid), or a concatenated tensor if
        ``self.cat_activations`` is set to ``True``.

        Example usage:

        .. code-block:: python

            activation_reader = ActivationReader(
                dir, activation_names=[(0, "hx"), (1, "hx")], **kwargs
            )

            # activation_name must be passed because ActivationReader
            # contains two activation_names.
            activations_first_sen = activation_reader[0, (1, "hx")]
            all_activations = activation_reader[:, (1, "hx")]


            activation_reader2 = ActivationReader(
                dir, activation_names=[(1, "hx")], **kwargs
            )

            # activation_name can be left implicit.
            activations_first_10_sens = activation_reader2[:10]

        Parameters
        ----------
        key : ActivationKey
            ``ActivationIndex`` or ``(index, activation_name)``, as
            explained above.

        Returns
        -------
        split_activations : Tensor | Iterator[Tensor, ...]
            Tensor, if ``self.cat_activations`` is set to True.
            Otherwise a Generator of tensors, with each item
            corresponding to the extracted activations of a specific
            sentence.

        .. automethod:: __getitem__
        """
        if isinstance(key, tuple):
            index, activation_name = key
        else:
            assert (
                len(self.activation_names) == 1
            ), "Activation name must be provided if multiple activations have been extracted"
            index = key
            activation_name = self.activation_names[0]

        iterable_index = activation_index_to_iterable(
            index, len(self.activation_ranges)
        )
        ranges = [self.activation_ranges[idx] for idx in iterable_index]

        sen_indices = torch.cat([torch.arange(*r) for r in ranges]).to(torch.long)

        if activation_name not in self.activation_dict:
            self._set_activations(activation_name)

        if self.cat_activations:
            return self.activation_dict[activation_name][sen_indices]

        return self.get_item_generator(ranges, self.activation_dict[activation_name])

    @staticmethod
    def get_item_generator(ranges, activations) -> Iterator[Tensor]:
        for start, stop in ranges:
            yield activations[start:stop]

    def __len__(self) -> int:
        """ Returns the total number of extracted activations. """
        return self.activation_ranges[-1][1]

    def to(self, device: str) -> None:
        """ Cast activations to a different device. """
        self.activation_dict = {
            a_name: activation.to(device)
            for a_name, activation in self.activation_dict.items()
        }

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
            self._set_activations(activation_name)

        return activations

    def _set_activations(self, activation_name: ActivationName) -> None:
        """Reads the pickled activations of activation_name

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
                            (len(self), hidden_size),
                            dtype=sen_activations.dtype,
                            device=sen_activations.device,
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

        if not self.store_multiple_activations:
            self.activation_dict = {}  # reset activation_dict

        self.activation_dict[activation_name] = activations
