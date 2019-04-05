from contextlib import ExitStack
from time import time
from typing import List, Optional, Tuple

import numpy as np
import torch

from rnnalyse.activations.activation_writer import ActivationWriter
from rnnalyse.activations.init_states import InitStates
from rnnalyse.models.language_model import LanguageModel
from rnnalyse.typedefs.corpus import Corpus, CorpusSentence, Labels
from rnnalyse.typedefs.extraction import ActivationRanges, SelectFunc
from rnnalyse.typedefs.activations import ActivationNames, FullActivationDict, PartialArrayDict
from rnnalyse.utils.paths import trim


class Extractor:
    """ Extracts all intermediate activations of a LM from a corpus.

    Only activations that are provided in activation_names will be
    stored in a pickle file. Each activation is written to its own file.

    Parameters
    ----------
    model : LanguageModel
        Language model that inherits from LanguageModel.
    corpus : Corpus
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
    corpus : Corpus
    activation_names : List[tuple[int, str]]
    output_dir: str
    init_lstm_states : FullActivationDict
        Initial embeddings that are loaded from file or set to zero.
    activation_writer : ActivationWriter
        Auxiliary class that writes activations to file.
    """
    def __init__(self,
                 model: LanguageModel,
                 corpus: Corpus,
                 activation_names: ActivationNames,
                 output_dir: str,
                 init_lstm_states_path: Optional[str] = None) -> None:
        self.model = model
        self.corpus = corpus

        self.activation_names: ActivationNames = activation_names
        self.output_dir = trim(output_dir)
        self.init_lstm_states: InitStates = InitStates(
            model.num_layers, model.hidden_size, init_lstm_states_path
        )

        self.activation_writer = ActivationWriter(output_dir, list(activation_names))

    # TODO: Allow batch input
    def extract(self,
                cutoff: int = -1,
                print_every: int = 10,
                dynamic_dumping: bool = True,
                selection_func: SelectFunc = lambda pos, token, labeled_sentence: True,
                create_label_file: bool = True,
                create_avg_eos: bool = False) -> None:
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
        create_label_file : bool, optional
            Indicates whether to create a label file, defaults to True.
        create_avg_eos : bool, optional
            Toggle to save average end of sentence activations. Will be
            stored in in `self.output_dir`.
        """
        start_t = prev_t = time()
        tot_extracted = n_sens = 0

        all_activations: PartialArrayDict = self._init_sen_activations()
        activation_ranges: ActivationRanges = {}

        print(f'\nStarting extraction of {len(self.corpus) if cutoff < 0 else cutoff} sentences...')

        with ExitStack() as stack:
            self.activation_writer.create_output_files(stack, create_label_file, create_avg_eos)

            extracted_labels: Labels = []
            avg_eos_states = self._init_avg_eos_activations()

            for n_sens, (sen_id, labeled_sentence) in enumerate(self.corpus.items()):
                if n_sens % print_every == 0 and n_sens > 0:
                    self._print_time_info(prev_t, start_t, print_every, n_sens)
                    prev_t = time()
                if cutoff == n_sens:
                    break

                sen_activations, sen_extracted_labels, n_extracted = \
                    self._extract_sentence(labeled_sentence, selection_func)

                if dynamic_dumping:
                    self.activation_writer.dump_activations(sen_activations)
                else:
                    for name in all_activations.keys():
                        all_activations[name].append(sen_activations[name])

                if create_avg_eos:
                    self._update_avg_eos_activations(avg_eos_states, sen_activations)

                extracted_labels.extend(sen_extracted_labels)
                activation_ranges[sen_id] = (tot_extracted, tot_extracted+n_extracted)
                tot_extracted += n_extracted

            self.activation_writer.dump_activation_ranges(activation_ranges)
            if extracted_labels:
                self.activation_writer.dump_labels(extracted_labels)
            if create_avg_eos:
                self._normalize_avg_eos_activations(avg_eos_states, n_sens+1)
                self.activation_writer.dump_avg_eos(avg_eos_states)

            if not dynamic_dumping:
                for name in all_activations.keys():
                    all_activations[name] = np.concatenate(all_activations[name], axis=0)
                self.activation_writer.dump_activations(all_activations)

        if dynamic_dumping:
            print('\nConcatenating sequentially dumped pickle files into 1 array...')
            self.activation_writer.concat_pickle_dumps()

        minutes, seconds = divmod(time() - start_t, 60)

        print(f'\nExtraction finished.')
        print(f'{n_sens+1} sentences have been extracted, '
              f'yielding {tot_extracted} data points.')
        print(f'Total time took {minutes:.0f}m {seconds:.1f}s')

    @staticmethod
    def _print_time_info(prev_t: float, start_t: float, print_every: int, n_sens: int) -> None:
        speed = 1 / ((time() - prev_t) / print_every)
        duration = time() - start_t
        minutes, seconds = divmod(duration, 60)

        print(f'#sens: {n_sens:>4}\t\t'
              f'Time: {minutes:>3.0f}m {seconds:>2.1f}s\t'
              f'Speed: {speed:.2f}sen/s')

    def _extract_sentence(self,
                          sentence: CorpusSentence,
                          selection_func: SelectFunc) -> Tuple[PartialArrayDict, List, int]:
        """ Generates the embeddings of a sentence and writes to file.

        Parameters
        ----------
        sentence : CorpusSentence
            Corpus sentence containing the raw sentence and other info
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
        n_extracted = 0

        activations: FullActivationDict = self.init_lstm_states.create()

        for i, token in enumerate(sentence.sen):
            _out, activations = self.model(token, activations)

            # Check whether current activations match criterion defined in selection_func
            if selection_func(i, token, sentence):
                if sentence.labels is not None:
                    extracted_labels.append(sentence.labels[i])

                for layer, name in self.activation_names:
                    sen_activations[(layer, name)].append(
                        activations[layer][name].detach().numpy()
                    )

                n_extracted += 1

        for a_name, arr in sen_activations.items():
            sen_activations[a_name] = np.array(arr)

        return sen_activations, extracted_labels, n_extracted

    def _init_sen_activations(self, sen_len: int = 0) -> PartialArrayDict:
        """ Initialize dict of numpy arrays that will be written to file. """
        return {
            (layer, name): [] if sen_len == 0 else np.empty((sen_len, self.model.hidden_size))
            for (layer, name) in self.activation_names
        }

    def _init_avg_eos_activations(self) -> FullActivationDict:
        init_avg_eos_activations: FullActivationDict = {}

        for layer in range(self.model.num_layers):
            init_avg_eos_activations[layer] = {
                'hx': torch.zeros(self.model.hidden_size),
                'cx': torch.zeros(self.model.hidden_size),
            }
            if (layer, 'hx') not in self.activation_names:
                self.activation_names.append((layer, 'hx'))
            if (layer, 'cx') not in self.activation_names:
                self.activation_names.append((layer, 'cx'))

        return init_avg_eos_activations

    @staticmethod
    def _update_avg_eos_activations(prev_activations: FullActivationDict,
                                    new_activations: PartialArrayDict) -> None:
        for layer in prev_activations.keys():
            for name in prev_activations[layer].keys():
                eos_activation = new_activations[(layer, name)][-1]
                prev_activations[layer][name] += torch.from_numpy(eos_activation)

    @staticmethod
    def _normalize_avg_eos_activations(avg_eos_activations: FullActivationDict,
                                       n_sens: int) -> None:
        for layer in avg_eos_activations.keys():
            for name in avg_eos_activations[layer].keys():
                avg_eos_activations[layer][name] /= n_sens
