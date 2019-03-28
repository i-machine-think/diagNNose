import os
import pickle
import warnings
from contextlib import ExitStack
from typing import BinaryIO, Optional

import numpy as np

from rnnalyse.typedefs.activations import (
    ActivationFiles, ActivationNames, FullActivationDict, PartialArrayDict)
from rnnalyse.typedefs.corpus import Labels
from rnnalyse.typedefs.extraction import ActivationRanges
from rnnalyse.utils.paths import dump_pickle, trim

from .activation_reader import ActivationReader


class ActivationWriter:
    """ Writes activations to file, using an ExitStack.

    Parameters
    ----------
    output_dir : str, optional
        Directory to which activations will be written
    activation_names : List[tuple[int, str]]
        List of (layer, activation_name) tuples

    Attributes
    ----------
    output_dir : str
    activation_files : ActivationFiles
        Dict of files to which activations will be written.
    label_file: Optional[BinaryIO]
        File to which sentence labels will be written.
    avg_eos_file: Optional[BinaryIO]
        File to which avg end of sentence activations will be written
    """
    def __init__(self, output_dir: str, activation_names: ActivationNames) -> None:
        self.output_dir = trim(output_dir)
        self.activation_names = activation_names

        self.activation_files: ActivationFiles = {}
        self.activation_ranges_file: Optional[BinaryIO] = None
        self.label_file: Optional[BinaryIO] = None
        self.avg_eos_file: Optional[BinaryIO] = None

    def create_output_files(self,
                            stack: ExitStack,
                            create_label_file: bool = True,
                            create_avg_eos_file: bool = False) -> None:
        """ Opens a file for each to-be-extracted activation. """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        # check if output directory is empty
        if os.listdir(self.output_dir):
            warnings.warn("Output directory %s is not empty" % self.output_dir)

        self.activation_files = {
            (layer, name):
                stack.enter_context(
                    open(f'{self.output_dir}/{name}_l{layer}.pickle', 'wb')
                )
            for (layer, name) in self.activation_names
        }
        self.activation_ranges_file = stack.enter_context(
            open(f'{self.output_dir}/ranges.pickle', 'wb')
        )
        if create_label_file:
            self.label_file = stack.enter_context(
                open(f'{self.output_dir}/labels.pickle', 'wb')
            )
        if create_avg_eos_file:
            self.avg_eos_file = stack.enter_context(
                open(f'{self.output_dir}/avg_eos.pickle', 'wb')
            )

    def dump_activations(self, activations: PartialArrayDict) -> None:
        """ Dumps the generated activations to a list of opened files

        Parameters
        ----------
        activations : PartialActivationDict
            The Tensors of each activation that was specifed by
            self.activation_names at initialization.
        """
        for layer, name in self.activation_names:
            assert (layer, name) in self.activation_files.keys(), 'Activation file is not opened'
            pickle.dump(activations[(layer, name)], self.activation_files[(layer, name)])

    def dump_activation_ranges(self, activation_ranges: ActivationRanges) -> None:
        assert self.activation_ranges_file is not None

        pickle.dump(activation_ranges, self.activation_ranges_file)

    def dump_labels(self, extracted_labels: Labels) -> None:
        assert self.label_file is not None

        labels: np.ndarray = np.array(extracted_labels)

        pickle.dump(labels, self.label_file)

    def dump_avg_eos(self, avg_eos_states: FullActivationDict) -> None:
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
        activation_reader = ActivationReader(self.output_dir)

        for (layer, name) in self.activation_names:
            activations = activation_reader.read_activations((layer, name))
            filename = f'{self.output_dir}/{name}_l{layer}.pickle'
            if not overwrite:
                filename = filename.replace('.pickle', '_concat.pickle')
            dump_pickle(activations, filename)
