import os
import pickle
import warnings
from contextlib import ExitStack
from typing import BinaryIO, Optional

from diagnnose.typedefs.activations import (
    ActivationFiles,
    ActivationNames,
    ActivationRanges,
    ActivationTensors,
)
from diagnnose.utils.pickle import dump_pickle

from .activation_reader import ActivationReader


class ActivationWriter:
    """ Writes activations to file, using an ExitStack.

    Parameters
    ----------
    activations_dir : str, optional
        Directory to which activations will be written

    Attributes
    ----------
    activations_dir : str
    activation_names : List[tuple[int, str]]
        List of (layer, activation_name) tuples
    activation_files : ActivationFiles
        Dict of files to which activations will be written.
    avg_eos_file: Optional[BinaryIO]
        File to which avg end of sentence activations will be written
    """

    def __init__(self, activations_dir: str) -> None:
        self.activations_dir = activations_dir

        self.activation_names: ActivationNames = []
        self.activation_files: ActivationFiles = {}
        self.activation_ranges_file: Optional[BinaryIO] = None
        self.avg_eos_file: Optional[BinaryIO] = None

    def create_output_files(
        self,
        stack: ExitStack,
        activation_names: ActivationNames,
        dump_activations: bool = True,
        dump_avg_eos: bool = False,
    ) -> None:
        """ Opens a file for each to-be-extracted activation. """
        self.activation_names = activation_names

        if not os.path.exists(self.activations_dir):
            os.makedirs(self.activations_dir)

        if os.listdir(self.activations_dir):
            warnings.warn("Output directory %s is not empty" % self.activations_dir)

        if dump_activations:
            self.activation_files = {
                (layer, name): stack.enter_context(
                    open(
                        os.path.join(self.activations_dir, f"{name}_l{layer}.pickle"),
                        "wb",
                    )
                )
                for (layer, name) in self.activation_names
            }
            self.activation_ranges_file = stack.enter_context(
                open(os.path.join(self.activations_dir, "ranges.pickle"), "wb")
            )
        if dump_avg_eos:
            self.avg_eos_file = stack.enter_context(
                open(os.path.join(self.activations_dir, "avg_eos.pickle"), "wb")
            )

    def dump_activations(self, activations: ActivationTensors) -> None:
        """ Dumps the generated activations to a list of opened files

        Parameters
        ----------
        activations : PartialArrayDict
            The Tensors for each activation that was specifed by
            self.activation_names at initialization.
        """
        for activation_name in self.activation_names:
            assert (
                activation_name in self.activation_files.keys()
            ), "Activation file is not opened"
            pickle.dump(
                activations[activation_name], self.activation_files[activation_name]
            )

    def dump_activation_ranges(self, activation_ranges: ActivationRanges) -> None:
        assert self.activation_ranges_file is not None

        pickle.dump(activation_ranges, self.activation_ranges_file)

    def dump_avg_eos(self, avg_eos_states: ActivationTensors) -> None:
        assert self.avg_eos_file is not None

        pickle.dump(avg_eos_states, self.avg_eos_file)

    def concat_pickle_dumps(self, overwrite: bool = True) -> None:
        """ Concatenates a sequential pickle dump and pickles to file .

        Note that this overwrites the sequential pickle dump by default.

        Parameters
        ----------
        overwrite : bool, optional
            Set to True to overwrite the file containing the sequential
            pickle dump, otherwise creates a new file. Defaults to True.
        """
        activation_reader = ActivationReader(self.activations_dir)

        for (layer, name) in self.activation_names:
            activations = activation_reader.read_activations((layer, name))
            filename = os.path.join(self.activations_dir, f"{name}_l{layer}.pickle")
            if not overwrite:
                filename = filename.replace(".pickle", "_concat.pickle")
            dump_pickle(activations, filename)
            del activations
