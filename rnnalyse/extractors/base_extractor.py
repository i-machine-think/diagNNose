import pickle
from contextlib import ExitStack
from time import time
from typing import BinaryIO, List, Optional, Callable
import os
import warnings

import numpy as np
import torch

from ..activations.initial import InitStates
from ..models.language_model import LanguageModel
from ..typedefs.corpus import LabeledCorpus, Labels, LabeledSentence
from ..typedefs.models import (
    ActivationFiles, ActivationName, FullActivationDict, PartialActivationDict)
from ..utils.paths import trim, dump_pickle


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
                 init_lstm_states_path: str = '') -> None:
        self.model = model
        self.corpus = corpus

        self.activation_names: List[ActivationName] = activation_names
        self.output_dir = os.path.expanduser(trim(output_dir))

        self.activation_files: ActivationFiles = {}
        self.label_file: Optional[BinaryIO] = None

        self.init_lstm_states: InitStates = InitStates(self.model, init_lstm_states_path)
        self.cur_time = time()
        self.num_extracted = 0
        self.n_sens = 0

    # TODO: Allow batch input
    def extract(self, cutoff: int = -1, print_every: int = 10,
                selection_func: Callable = lambda pos, token, labeled_sentence: True) -> None:
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
        selection_func: Callable
            Function which determines if activations for a token should be extracted or not.
        """
        start_time: float = time()
        print('\nStarting extraction...')

        with ExitStack() as stack:
            self._create_output_files(stack)

            for labeled_sentence in self.corpus.values():

                sen_activations = self._extract_sentence(labeled_sentence, selection_func)
                sen_num_extracted = list(sen_activations.values())[0].shape[0]

                self._dump_activations(sen_activations)
                self.num_extracted += sen_num_extracted
                self.n_sens += 1

                if self.n_sens % print_every == 0 and self.n_sens > 0:
                    self._print_time_info(start_time, print_every)
                if cutoff == self.n_sens:
                    break

            # TODO: Move this to separate Labeler class
            self._dump_static_info()

        minutes, seconds = divmod(time() - start_time, 60)

        print(f'\nExtraction finished.')
        print(f'{self.n_sens} sentences have been extracted, '
              f'yielding {self.num_extracted} data points.')
        print(f'Total time took {minutes:.0f}m {seconds:.2f}s')

    def _create_output_files(self, stack: ExitStack) -> None:
        """ Opens a file for each to-be-extracted activation. """
        # check if output directory is empty
        if os.listdir(self.output_dir): warnings.warn("Output directory %s is not empty" % self.output_dir)
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

    def _print_time_info(self, start_time: float, print_every: int) -> None:
        speed = (time() - self.cur_time) / print_every
        self.cur_time = time()
        duration = self.cur_time - start_time
        minutes, seconds = divmod(duration, 60)

        print(f'#sens: {self.n_sens:>4}\t\t'
              f'Time: {minutes:>3.0f}m {seconds:>2.2f}s\t'
              f'Speed: {speed:.2f}s/sen')

    def _extract_sentence(self, sentence: LabeledSentence, selection_func: Callable) -> PartialActivationDict:
        """ Generates the embeddings of a sentence and writes to file.

        Parameters
        ----------
        sentence : Sentence
            The to-be-extracted sentence, represented as a list of strings.
        selection_func: Callable
            Function which determines if activations for a token should be extracted or not.

        Returns
        -------
        num_extracted: int
            Number of extracted activations for this sentence.
        """

        sen_activations: PartialActivationDict = self._init_sen_activations(0)

        activations: FullActivationDict = self.init_lstm_states.states

        for i, token in enumerate(sentence.sen):
            out, activations = self.model(token, activations)

            # Check whether current activations match criterion defined in selection_func
            if selection_func(i, token, sentence):
                for layer, name in self.activation_names:
                    sen_activations[(layer, name)] = np.append(
                        sen_activations[(layer, name)], activations[layer][name].detach().numpy()[np.newaxis, ...],
                        axis=0
                    )

        return sen_activations

    def extract_average_eos_activations(self):
        """ Extract average end of sentence activations and dump them to a file. """

        def _incremental_avg(old_avg: torch.Tensor, new_value: torch.Tensor, n_sens: int):
            return old_avg + 1 / n_sens * (new_value - old_avg)

        def _eos_selection_func(pos: int, token: str, sentence: LabeledSentence):
            return pos == len(sentence.sen) - 1

        # Init states
        avg_eos_states = {
            layer: {'hx': torch.zeros(self.model.hidden_size), 'cx': torch.zeros(self.model.hidden_size)}
            for layer in range(self.model.num_layers)
        }

        # Extract
        for i, labeled_sentence in enumerate(self.corpus.values()):
            eos_activations = self._extract_sentence(labeled_sentence, _eos_selection_func)

            # Update average eos states if last activation was extracted
            avg_eos_states = {
                layer: {
                    'hx': _incremental_avg(
                        avg_eos_states[layer]['hx'], torch.Tensor(eos_activations[(layer, 'hx')]), n_sens=i+1
                    ),
                    'cx': _incremental_avg(
                        avg_eos_states[layer]['cx'], torch.Tensor(eos_activations[(layer, 'cx')]), n_sens=i+1
                    )
                }
                for layer in avg_eos_states
            }

        # Dump average eos states
        dump_pickle(avg_eos_states, f'{self.output_dir}/avg_eos.pickle')

    def _dump_activations(self, activations: PartialActivationDict) -> None:
        """ Dumps the generated activations to a list of opened files

        Parameters
        ----------
        activations : PartialActivationDict
            The Tensors of each activation that was specifed by
            self.activation_names at initialization.
        """
        for layer, name in self.activation_names:
            pickle.dump(activations[(layer, name)], self.activation_files[(layer, name)])

    def _dump_static_info(self) -> None:
        assert self.label_file is not None

        labels: Labels = np.array([label
                                   for sen in self.corpus.values()
                                   for label in sen.labels
                                   ][:self.num_extracted])

        pickle.dump(labels, self.label_file)

    def _init_sen_activations(self, sen_len: int) -> PartialActivationDict:
        """ Initialize dict of Tensors that will be written to file. """
        return {
            (layer, name): np.empty((sen_len, self.model.hidden_size))
            for (layer, name) in self.activation_names
        }
