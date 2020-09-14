import shutil
from typing import Optional, Tuple

from diagnnose.activations import ActivationReader
from diagnnose.activations.selection_funcs import return_all
from diagnnose.corpus import Corpus
from diagnnose.extract import BATCH_SIZE, Extractor
from diagnnose.typedefs.activations import (
    ActivationNames,
    RemoveCallback,
    SelectionFunc,
)
from diagnnose.utils.misc import suppress_print


@suppress_print
def simple_extract(
    model: "LanguageModel",
    corpus: Corpus,
    activation_names: ActivationNames,
    activations_dir: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    selection_func: SelectionFunc = return_all,
    sen_column: str = "sen",
) -> Tuple[ActivationReader, RemoveCallback]:
    """Basic extraction method.

    Returns
    -------
    remove_activations : RemoveCallback
        callback function that can be executed at the end of a procedure
        that depends on the extracted activations. Removes all the
        activations that have been extracted. Takes no arguments.
    """
    corpus.sen_column = sen_column

    extractor = Extractor(
        model,
        corpus,
        activation_names,
        activations_dir=activations_dir,
        batch_size=batch_size,
        selection_func=selection_func,
    )

    activation_reader = extractor.extract()

    def remove_activations():
        if activations_dir is not None:
            shutil.rmtree(activations_dir)

    return activation_reader, remove_activations
