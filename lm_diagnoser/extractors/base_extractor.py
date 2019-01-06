from typing import BinaryIO, List, Optional
from customtypes.models import ActivationName, ActivationFiles, PartialActivationDict
from customtypes.corpus import LabeledCorpus, LabeledSentence, Labels

from corpus.import_corpus import convert_to_labeled_corpus
from embeddings.initial import InitEmbs
from models.language_model import LanguageModel

import torch
import pickle
from time import time
from contextlib import ExitStack


class Extractor:
    """ Extracts all intermediate activations of a LM from a corpus.

    Only activations that are provided in activation_names will be
    stored in a pickle file. Each activation is written to its own file.

    Parameters
    ----------
    model : LanguageModel
        Language model that inherits from LanguageModel.
    corpus_path : str
        Location of labeled corpus.
    activation_names : List[ActivationName]
        List of activations to be stored of the form [(l, n)] with
            l (int): the layer number.
            n (str): activation name as defined in model.
    init_embs_path : str, optional (default = '')
        Location of initial embeddings.

    Attributes
    ----------
    model : LanguageModel
        Language model that inherits from LanguageModel.
    hidden_size : int
        Number of hidden units in model.
    activation_names : List[ActivationName]
        List of activations to be stored.
    activation_files : ActivationFiles
        Dict of files to which activations will be written.
    label_file: Optional[BinaryIO]
        File to which sentence labels will be written.
    corpus : LabeledCorpus
        corpus containing the labels for each sentence.
    init_embs : FullActivationDict
        initial embeddings that are loaded from file or set to zero.
    """
    def __init__(self,
                 model: LanguageModel,
                 corpus_path: str,
                 activation_names: List[ActivationName],
                 init_embs_path: str = '') -> None:

        self.model = model
        self.hidden_size = model.hidden_size
        self.activation_names = activation_names
        self.activation_files: ActivationFiles = {}
        self.label_file: Optional[BinaryIO] = None

        self.corpus: LabeledCorpus = convert_to_labeled_corpus(corpus_path)

        self.init_embs = InitEmbs(init_embs_path, model).activations

    # TODO: Allow batch input
    def extract(self,
                output_path: str,
                cutoff: int = -1,
                print_every: int = 100) -> None:
        """ Extracts embeddings from a labeled corpus.

        Uses contextlib.ExitStack to write to multiple files at once.
        File writing is done directly per sentence, to lessen RAM usage.

        Parameters
        ----------
        output_path : str
            Directory to which extractions will be saved.
        cutoff : int, optional (default = -1)
            Number of sentences to be extracted. Set to -1 to extract
            entire corpus.
        print_every : int, optional (default = 100)
            Print progress info every n steps.
        """
        start_time: float = time()

        with ExitStack() as stack:
            self._create_output_files(output_path, stack)

            n_sens = 0
            num_extracted = 0

            for labeled_sentence in self.corpus:
                if n_sens % print_every == 0 and n_sens > 0:
                    print(f'{n_sens}\t{time() - start_time:.2f}s')

                self._extract_sentence(labeled_sentence)

                num_extracted += len(labeled_sentence.sen)
                n_sens += 1

                if cutoff == n_sens:
                    break

        print(f'\nExtraction finished.')
        print(f'{n_sens} sentences have been extracted, yielding {num_extracted} data points.')
        print(f'Total time took {time() - start_time:.2f}s')

    def _create_output_files(self, output_path: str, stack: ExitStack) -> None:
        """ Opens a file for each to-be-extracted activation. """
        self.activation_files = {
            (l, name):
                stack.enter_context(
                    open(f'{output_path}/{name}_l{l}.pickle', 'wb')
                )
            for (l, name) in self.activation_names
        }
        self.label_file = stack.enter_context(
            open(f'{output_path}/labels.pickle', 'wb')
        )

    def _extract_sentence(self, labeled_sentence: LabeledSentence) -> None:
        labeled_sentence.validate()

        sen_activations = self._init_sen_activations(len(labeled_sentence))

        activations = self.init_embs

        for i, token in enumerate(labeled_sentence.sen):
            out, activations = self.model(token, activations)

            for l, name in self.activation_names:
                sen_activations[(l, name)][i] = activations[l][name]

        self._dump_activations(sen_activations, labeled_sentence.labels)

    def _dump_activations(self,
                          activations: PartialActivationDict,
                          labels: Labels) -> None:
        for l, name in self.activation_names:
            pickle.dump(activations[(l, name)], self.activation_files[(l, name)])

        assert self.label_file is not None
        pickle.dump(labels, self.label_file)

    def _init_sen_activations(self, sen_len: int) -> PartialActivationDict:
        return {
            (l, name): torch.zeros(sen_len, self.hidden_size)
            for (l, name) in self.activation_names
        }
