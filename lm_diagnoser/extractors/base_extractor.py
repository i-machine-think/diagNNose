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
    corpus : LabeledCorpus
    activation_names : List[tuple[int, str]]
    output_dir: str, optional
    activation_files : ActivationFiles
        Dict of files to which activations will be written.
    label_file: Optional[BinaryIO]
        File to which sentence labels will be written.
    init_lstm_states : FullActivationDict
        Initial embeddings that are loaded from file or set to zero.
    num_extracted : int
        Current amount of extracted activations, incremented once per w.
    n_sens : int
        Current amount of extracted sentences.
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
        self.output_dir = trim(output_dir)

        self.activation_files: ActivationFiles = {}
        self.label_file: Optional[BinaryIO] = None

        self.init_lstm_states: InitStates = InitStates(init_lstm_states_path, self.model)
        self.num_extracted = 0
        self.n_sens = 0

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
        cur_time: float = start_time
        print('\nStarting extraction...')

        with ExitStack() as stack:
            self._create_output_files(stack)

            for labeled_sentence in self.corpus.values():
                if self.n_sens % print_every == 0 and self.n_sens > 0:
                    self._print_time_info(start_time, cur_time, print_every)

                self._extract_sentence(labeled_sentence.sen)

                self.num_extracted += len(labeled_sentence.sen)
                self.n_sens += 1

                if cutoff == self.n_sens:
                    break

            # TODO: Move this to separate Labeler class
            self._dump_static_info()

        print(f'\nExtraction finished.')
        print(f'{self.n_sens} sentences have been extracted, '
              f'yielding {self.num_extracted} data points.')
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
        self.label_file = stack.enter_context(
            open(f'{self.output_dir}/labels.pickle', 'wb')
        )

    def _print_time_info(self, start_time: float, cur_time: float, print_every: int) -> None:
        speed = (time() - cur_time) / print_every
        cur_time = time()
        time_passed = cur_time - start_time
        print(f'#sens: {self.n_sens}\t'
              f'Time passed:{time_passed:.1f}s\t'
              f'Speed:{speed:.1f}s/sen')

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
