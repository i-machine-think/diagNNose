from diagnnose.extractors.base_extractor import Extractor
from diagnnose.models.lm import LanguageModel
from diagnnose.typedefs.activations import ActivationNames, SelectFunc
from diagnnose.typedefs.corpus import Corpus
from diagnnose.utils.misc import suppress_print

BATCH_SIZE = 1024


@suppress_print
def simple_extract(
    model: LanguageModel,
    activations_dir: str,
    corpus: Corpus,
    activation_names: ActivationNames,
    selection_func: SelectFunc = lambda sen_id, pos, item: True,
) -> None:
    """ Basic extraction method. """
    extractor = Extractor(model, corpus, activations_dir, activation_names)

    extractor.extract(
        batch_size=BATCH_SIZE, dynamic_dumping=False, selection_func=selection_func
    )
