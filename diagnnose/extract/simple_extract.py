import shutil
from typing import TYPE_CHECKING, Optional, Tuple

from diagnnose.activations import ActivationReader
from diagnnose.activations.selection_funcs import return_all
from diagnnose.corpus import Corpus
from diagnnose.typedefs.activations import (
    ActivationNames,
    RemoveCallback,
    SelectionFunc,
)
from diagnnose.utils.misc import suppress_print

if TYPE_CHECKING:
    from diagnnose.models import LanguageModel

from .extractor import BATCH_SIZE, Extractor


@suppress_print
def simple_extract(
    model: "LanguageModel",
    corpus: Corpus,
    activation_names: ActivationNames,
    activations_dir: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    selection_func: SelectionFunc = return_all,
) -> Tuple[ActivationReader, RemoveCallback]:
    """Basic extraction method.

    Parameters
    ----------
    model : LanguageModel
        Language model that inherits from LanguageModel.
    corpus : Corpus
        Corpus containing sentences to be extracted.
    activation_names : List[tuple[int, str]]
        List of (layer, activation_name) tuples
    activations_dir : str, optional
        Directory to which activations will be written. If not provided
        the `extract()` method will only return the activations without
        writing them to disk.
    selection_func : SelectionFunc
        Function which determines if activations for a token should
        be extracted or not.
    batch_size : int, optional
        Amount of sentences processed per forward step. Higher batch
        size increases extraction speed, but should be done
        accordingly to the amount of available RAM. Defaults to 1.

    Returns
    -------
    activation_reader : ActivationReader
        ActivationReader for the activations that have been extracted.
    remove_activations : RemoveCallback
        Callback function that can be executed at the end of a procedure
        that depends on the extracted activations. Removes all the
        activations that have been extracted. Takes no arguments.
    """
    extractor = Extractor(
        model,
        corpus,
        activation_names,
        activations_dir=activations_dir,
        selection_func=selection_func,
        batch_size=batch_size,
    )

    activation_reader = extractor.extract()

    def remove_activations():
        if activations_dir is not None:
            shutil.rmtree(activations_dir)

    return activation_reader, remove_activations
