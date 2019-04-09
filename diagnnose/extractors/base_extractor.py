from contextlib import ExitStack
from time import time
from typing import List, Optional, Tuple

import numpy as np
import torch

from ..activations.activation_writer import ActivationWriter
from ..activations.initial import InitStates
from ..models.language_model import LanguageModel
from ..typedefs.corpus import LabeledCorpus, LabeledSentence, Labels
from ..typedefs.extraction import ActivationRanges, SelectFunc
from ..typedefs.activations import ActivationNames, FullActivationDict, PartialArrayDict
from ..utils.paths import dump_pickle, trim


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
    output_dir : str
        Directory to which activations will be written
    init_lstm_states_path: str, optional
        Path to pickled initial embeddings

    Attributes
    ----------
    model : LanguageModel
    corpus : LabeledCorpus
    activation_names : List[tuple[int, str]]
    output_dir: str
    init_lstm_states : FullActivationDict
        Initial embeddings that are loaded from file or set to zero.
    activation_writer : ActivationWriter
        Auxiliary class that writes activations to file.
    """
    def __init__(self,
                 model: LanguageModel,
                 corpus: LabeledCorpus,
                 activation_names: ActivationNames,
                 output_dir: str,
                 init_lstm_states_path: Optional[str] = None) -> None:
        self.model = model
        self.corpus = corpus

        self.activation_names: ActivationNames = activation_names
        self.output_dir = trim(output_dir)
        self.init_lstm_states: InitStates = InitStates(model, init_lstm_states_path)

        self.activation_writer = ActivationWriter(output_dir, activation_names)

    # TODO: Allow batch input
    def extract(self,
                cutoff: int = -1,
                print_every: int = 10,
                dynamic_dumping: bool = True,
                selection_func: SelectFunc = lambda pos, token, labeled_sentence: True) -> None:
        """ Extracts embeddings from a labeled corpus.

        Uses contextlib.ExitStack to write to multiple files at once.
        File writing is done directly per sentence, to lessen RAM usage.

        Parameters
        ----------
        cutoff: int, optional
            How many sentences of the corpus to extract activations for.
            Setting this parameter to -1 will extract the entire corpus,
            otherwise extraction is halted after extracting n sentences.
        print_every : int, optional
            Print time passed every n sentences, defaults to 10.
        dynamic_dumping : bool, optional
            Dump files dynamically, i.e. once per sentence, or dump
            all files at the end of extraction. Defaults to True.
        selection_func: Callable
            Function which determines if activations for a token should
            be extracted or not.
        """
        start_t = prev_t = time()
        n_extracted = n_sens = 0
        all_activations: PartialArrayDict = self._init_sen_activations()
        activation_ranges: ActivationRanges = {}
        print('\nStarting extraction...')

        with ExitStack() as stack:
            self.activation_writer.create_output_files(stack)
            extracted_labels: Labels = []

            for n_sens, (sen_id, labeled_sentence) in enumerate(self.corpus.items()):
                if n_sens % print_every == 0 and n_sens > 0:
                    self._print_time_info(prev_t, start_t, print_every, n_sens)
                    prev_t = time()
                if cutoff == n_sens:
                    break

                sen_activations, sen_extracted_labels = self._extract_sentence(labeled_sentence,
                                                                               selection_func)
                if dynamic_dumping:
                    self.activation_writer.dump_activations(sen_activations)
                else:
                    for name in all_activations.keys():
                        all_activations[name].append(sen_activations[name])

                extracted_labels.extend(sen_extracted_labels)
                activation_ranges[sen_id] = (n_extracted, n_extracted+len(sen_extracted_labels))
                n_extracted += len(sen_extracted_labels)

            self.activation_writer.dump_labels(extracted_labels)
            self.activation_writer.dump_activation_ranges(activation_ranges)
            if not dynamic_dumping:
                for name in all_activations.keys():
                    all_activations[name] = np.concatenate(all_activations[name], axis=0)
                self.activation_writer.dump_activations(all_activations)

        if dynamic_dumping:
            print('\nConcatenating sequentially dumped pickle files into 1 array...')
            self.activation_writer.concat_pickle_dumps()

        minutes, seconds = divmod(time() - start_t, 60)

        print(f'\nExtraction finished.')
        print(f'{n_sens} sentences have been extracted, '
              f'yielding {n_extracted} data points.')
        print(f'Total time took {minutes:.0f}m {seconds:.1f}s')

    @staticmethod
    def _print_time_info(prev_t: float, start_t: float, print_every: int, n_sens: int) -> None:
        speed = (time() - prev_t) / print_every
        duration = time() - start_t
        minutes, seconds = divmod(duration, 60)

        print(f'#sens: {n_sens:>4}\t\t'
              f'Time: {minutes:>3.0f}m {seconds:>2.1f}s\t'
              f'Speed: {speed:.2f}s/sen')

    def _extract_sentence(self,
                          sentence: LabeledSentence,
                          selection_func: SelectFunc) -> Tuple[PartialArrayDict, List]:
        """ Generates the embeddings of a sentence and writes to file.

        Parameters
        ----------
        sentence : Sentence
            To-be-extracted sentence, represented as a list of strings.
        selection_func : SelectFunc
            Function that determines whether activations for a token
            should be extracted or not.

        Returns
        -------
        sen_activations : PartialArrayDict
            Extracted activations for this sentence. Activations are
            converted to numpy arrays.
        extracted_labels : Labels
            List of labels corresponding to the extracted activations.
        """

        sen_activations: PartialArrayDict = self._init_sen_activations()
        extracted_labels: Labels = []

        activations: FullActivationDict = self.init_lstm_states.create()

        for i, token in enumerate(sentence.sen):
            _out, activations = self.model(token, activations)

            # Check whether current activations match criterion defined in selection_func
            if selection_func(i, token, sentence):
                extracted_labels.append(sentence.labels[i])

                for layer, name in self.activation_names:
                    sen_activations[(layer, name)].append(
                        activations[layer][name].detach().numpy()
                    )

        for a_name, arr in sen_activations.items():
            sen_activations[a_name] = np.array(arr)

        return sen_activations, extracted_labels

    # TODO: refactor
    def extract_average_eos_activations(self, print_every: int = 10) -> None:
        """ Extract average end of sentence activations and dump them to a file. """

        def _incremental_avg(old_avg: torch.Tensor,
                             new_value: torch.Tensor,
                             n_sens: int) -> torch.Tensor:
            return old_avg + 1 / n_sens * (new_value - old_avg)

        def _eos_selection_func(pos: int, _token: str, sentence: LabeledSentence) -> bool:
            return pos == (len(sentence.sen) - 1)

        start_time = prev_time = time()
        print('\nStarting extraction for average eos activations...')

        # Init states
        avg_eos_states = {
            layer: {
                'hx': torch.zeros(self.model.hidden_size),
                'cx': torch.zeros(self.model.hidden_size)
            }
            for layer in range(self.model.num_layers)
        }

        # Extract
        for i, labeled_sentence in enumerate(self.corpus.values()):
            eos_activations, _ = self._extract_sentence(labeled_sentence, _eos_selection_func)

            # Update average eos states if last activation was extracted
            avg_eos_states = {
                layer: {
                    'hx': _incremental_avg(
                        avg_eos_states[layer]['hx'],
                        torch.Tensor(eos_activations[(layer, 'hx')].squeeze(0)),
                        n_sens=i+1
                    ),
                    'cx': _incremental_avg(
                        avg_eos_states[layer]['cx'],
                        torch.Tensor(eos_activations[(layer, 'cx')].squeeze(0)),
                        n_sens=i+1
                    )
                }
                for layer in avg_eos_states
            }

            if i % print_every == 0 and i > 0:
                self._print_time_info(prev_time, start_time, print_every, i)
                prev_time = time()

        minutes, seconds = divmod(time() - start_time, 60)

        # Dump average eos states
        dump_pickle(avg_eos_states, f'{self.output_dir}/avg_eos.pickle')

        print(f'\nExtraction finished.')
        print(f'Total time took {minutes:.0f}m {seconds:.2f}s')

    def _init_sen_activations(self, sen_len: int = 0) -> PartialArrayDict:
        """ Initialize dict of numpy arrays that will be written to file. """
        return {
            (layer, name): [] if sen_len == 0 else np.empty((sen_len, self.model.hidden_size))
            for (layer, name) in self.activation_names
        }
