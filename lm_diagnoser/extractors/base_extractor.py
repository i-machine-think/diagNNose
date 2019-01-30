import pickle
from contextlib import ExitStack
from time import time
from typing import BinaryIO, List, Optional

import numpy as np

from ..activations.initial import InitStates
from ..models.language_model import LanguageModel
from ..typedefs.corpus import LabeledCorpus, Labels, Sentence
from ..typedefs.models import (
    ActivationFiles, ActivationName, FullActivationDict, PartialActivationDict)
from ..utils.paths import trim


class Extractor:
    """ Extracts all intermediate activations of a LM from a corpus.

    Only activations that are provided in activation_names will be
    stored in a pickle file. Each activation is written to its own file.

    Parameters
    ----------
    model : LanguageModel
        Language model that inherits from LanguageModel.
    corpus : LabeledCorpus
        Corpus containing the labels for each sentence.
    activation_names : List[tuple[int, str]]
        List of (layer, activation_name) tuples
    output_dir: str, optional
        Directory to which activations will be written
    init_lstm_states_path: str, optional
        Path to pickled initial embeddings

    Attributes
    ----------
    model : LanguageModel
        Language model that inherits from LanguageModel.
    corpus : LabeledCorpus
        Corpus containing the labels for each sentence.
    activation_names : List[ActivationName]
        List of activations to be stored.
    output_dir : str
        Directory to which activations will be written
    activation_files : ActivationFiles
        Dict of files to which activations will be written.
    label_file: Optional[BinaryIO]
        File to which sentence labels will be written.
    keys_file: Optional[BinaryIO]
        File to which sentence keys will be written.
    init_lstm_states : FullActivationDict
        Initial embeddings that are loaded from file or set to zero.
    """
    def __init__(self,
                 model: LanguageModel,
                 corpus: LabeledCorpus,
                 activation_names: List[ActivationName],
                 output_dir: str,
                 init_lstm_states_path: str = '',
                 ) -> None:
        self.model = model
        self.corpus = corpus

        self.activation_names: List[ActivationName] = activation_names
        self.output_dir = output_dir

        self.activation_files: ActivationFiles = {}
        self.label_file: Optional[BinaryIO] = None
        self.keys_file: Optional[BinaryIO] = None

        self.init_lstm_states: InitStates = InitStates(init_lstm_states_path, self.model)

    # TODO: Allow batch input
    def extract(self, cutoff: int = -1, print_every: int = 10) -> None:
        """ Extracts embeddings from a labeled corpus.

        Uses contextlib.ExitStack to write to multiple files at once.
        File writing is done directly per sentence, to lessen RAM usage.

        Parameters
        ----------
        cutoff: int, optional
            How many sentences of the corpus to extract activations for
            Setting this parameter to -1 will extract the entire corpus, 
            otherwise extraction is halted after extracting n sentences.
        print_every: int, optional
            Print time passed every n sentences, defaults to 10.
        """
        start_time: float = time()
        print('\nStarting extraction...')

        with ExitStack() as stack:
            self._create_output_files(stack)

            n_sens = 0
            num_extracted = 0

            for labeled_sentence in self.corpus.values():
                if n_sens % print_every == 0 and n_sens > 0:
                    print(f'{n_sens}\t{time() - start_time:.2f}s')

                self._extract_sentence(labeled_sentence.sen)

                num_extracted += len(labeled_sentence.sen)
                n_sens += 1

                if cutoff == n_sens:
                    break

            self._dump_static_info(n_sens, num_extracted)

        print(f'\nExtraction finished.')
        print(f'{n_sens} sentences have been extracted, yielding {num_extracted} data points.')
        print(f'Total time took {time() - start_time:.2f}s')

    def _create_output_files(self, stack: ExitStack) -> None:
        """ Opens a file for each to-be-extracted activation. """
        self.activation_files = {
            (l, name):
                stack.enter_context(
                    open(f'{self.output_dir}/{name}_l{l}.pickle', 'wb')
                )
            for (l, name) in self.activation_names
        }
        self.keys_file = stack.enter_context(
            open(f'{self.output_dir}/keys.pickle', 'wb')
        )
        self.label_file = stack.enter_context(
            open(f'{self.output_dir}/labels.pickle', 'wb')
        )

    def _extract_sentence(self, sentence: Sentence) -> None:
        """ Generates the embeddings of a sentence and writes to file.

        Parameters
        ----------
        sentence : Sentence
            The to-be-extracted sentence, represented as a list of strings.
        """
        sen_activations: PartialActivationDict = self._init_sen_activations(len(sentence))

        activations: FullActivationDict = self.init_lstm_states.states

        for i, token in enumerate(sentence):
            out, activations = self.model(token, activations)

            for l, name in self.activation_names:
                sen_activations[(l, name)][i] = activations[l][name].detach().numpy()

        self._dump_activations(sen_activations)

    def _dump_activations(self, activations: PartialActivationDict) -> None:
        """ Dumps the generated activations to a list of opened files

        Parameters
        ----------
        activations : PartialActivationDict
            The Tensors of each activation that was specifed by
            self.activation_names at initialization.
        """
        for l, name in self.activation_names:
            pickle.dump(activations[(l, name)], self.activation_files[(l, name)])

    def _dump_static_info(self, n_sens: int, num_extracted: int) -> None:
        assert self.keys_file is not None
        assert self.label_file is not None

        keys: List[int] = list(self.corpus.keys())[:n_sens]
        labels: Labels = np.array([label
                                   for sen in self.corpus.values()
                                   for label in sen.labels
                                   ][:num_extracted])

        pickle.dump(keys, self.keys_file)
        pickle.dump(labels, self.label_file)

    def _init_sen_activations(self, sen_len: int) -> PartialActivationDict:
        """ Initialize dict of Tensors that will be written to file. """
        return {
            (l, name): np.empty((sen_len, self.model.hidden_size))
            for (l, name) in self.activation_names
        }
