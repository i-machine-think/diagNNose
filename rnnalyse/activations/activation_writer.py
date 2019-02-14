import os
import pickle
import warnings
from contextlib import ExitStack
from typing import BinaryIO, Optional

import numpy as np

from ..typedefs.corpus import Labels
from ..typedefs.models import ActivationFiles, ActivationNames, PartialActivationDict
from ..utils.paths import trim


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

    """
    def __init__(self, output_dir: str, activation_names: ActivationNames) -> None:
        self.output_dir = trim(output_dir)
        self.activation_names = activation_names

        self.activation_files: ActivationFiles = {}
        self.label_file: Optional[BinaryIO] = None

    def create_output_files(self, stack: ExitStack) -> None:
        """ Opens a file for each to-be-extracted activation. """
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
        self.label_file = stack.enter_context(
            open(f'{self.output_dir}/labels.pickle', 'wb')
        )

    def dump_activations(self, activations: PartialActivationDict) -> None:
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

    def dump_labels(self, extracted_labels: list) -> None:
        assert self.label_file is not None

        labels: Labels = np.array(extracted_labels)

        pickle.dump(labels, self.label_file)
