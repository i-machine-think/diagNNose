from contextlib import ExitStack
from typing import List, Tuple

import numpy as np
import torch
from torchtext.data import Batch, Example
from tqdm import tqdm

from diagnnose.activations.activation_writer import ActivationWriter
from diagnnose.corpus.create_iterator import create_iterator
from diagnnose.typedefs.activations import (
    ActivationNames,
    BatchArrayDict,
    FullActivationDict,
    PartialArrayDict,
)
from diagnnose.typedefs.corpus import Corpus
from diagnnose.typedefs.extraction import ActivationRanges, SelectFunc
from diagnnose.typedefs.models import LanguageModel


class Extractor:
    """ Extracts all intermediate activations of a LM from a corpus.

    Only activations that are provided in activation_names will be
    stored in a pickle file. Each activation is written to its own file.

    Parameters
    ----------
    model : LanguageModel
        Language model that inherits from LanguageModel.
    corpus : Corpus
        Corpus containing sentences to be extracted.
    activations_dir : str
        Directory to which activations will be written

    Attributes
    ----------
    model : LanguageModel
    corpus : Corpus
    activation_names : List[tuple[int, str]]
    activations_dir : str
    activation_writer : ActivationWriter
        Auxiliary class that writes activations to file.
    """

    def __init__(
        self, model: LanguageModel, corpus: Corpus, activations_dir: str
    ) -> None:
        self.model = model
        self.corpus = corpus

        self.activation_names: ActivationNames = []

        self.activation_writer = ActivationWriter(activations_dir)

    # TODO: refactor
    def extract(
        self,
        activation_names: ActivationNames,
        batch_size: int = 1,
        cutoff: int = -1,
        dynamic_dumping: bool = True,
        selection_func: SelectFunc = lambda pos, token, item: True,
        create_avg_eos: bool = False,
        only_dump_avg_eos: bool = False,
    ) -> None:
        """ Extracts embeddings from a corpus.

        Uses contextlib.ExitStack to write to multiple files at once.
        File writing is done directly per sentence, to lessen RAM usage.

        Parameters
        ----------
        activation_names : List[tuple[int, str]]
            List of (layer, activation_name) tuples
        batch_size : int, optional
            Amount of sentences processed per forward step. Higher batch
            size increases extraction speed, but should be done
            accordingly to the amount of available RAM. Defaults to 1.
        cutoff : int, optional
            How many sentences of the corpus to extract activations for.
            Setting this parameter to -1 will extract the entire corpus,
            otherwise extraction is halted after extracting n sentences.
        dynamic_dumping : bool, optional
            Dump files dynamically, i.e. once per sentence, or dump
            all files at the end of extraction. Defaults to True.
        selection_func : Callable[[int, int, Example], bool]
            Function which determines if activations for a token should
            be extracted or not.
        create_avg_eos : bool, optional
            Toggle to save average end of sentence activations. Will be
            stored in in `self.output_dir`.
        only_dump_avg_eos : bool , optional
            Toggle to only save the average eos activations.
        """
        self.activation_names = activation_names

        tot_extracted = n_sens = 0

        all_activations: PartialArrayDict = self._init_sen_activations()
        activation_ranges: ActivationRanges = {}
        iterator = create_iterator(
            self.corpus, batch_size=batch_size, device=self.model.device
        )

        tot_num = len(self.corpus) if cutoff == -1 else cutoff
        print(f"\nStarting extraction of {tot_num} sentences...")

        with ExitStack() as stack:
            self.activation_writer.create_output_files(
                stack, activation_names, create_avg_eos, only_dump_avg_eos
            )

            if create_avg_eos:
                avg_eos_states = self._init_avg_eos_activations()

            for batch in tqdm(iterator, unit="batch"):
                batch_examples = self.corpus.examples[
                    n_sens : n_sens + batch.batch_size
                ]

                batch_activations, n_extracted = self._extract_sentence(
                    batch, batch_examples, selection_func
                )

                if not only_dump_avg_eos:
                    if dynamic_dumping:
                        for j in batch_activations.keys():
                            self.activation_writer.dump_activations(
                                batch_activations[j]
                            )
                    else:
                        for j in batch_activations.keys():
                            for name in all_activations.keys():
                                all_activations[name].append(batch_activations[j][name])

                if create_avg_eos:
                    self._update_avg_eos_activations(avg_eos_states, batch_activations)

                for j in batch_activations.keys():
                    activation_ranges[n_sens] = (
                        tot_extracted,
                        tot_extracted + n_extracted[j],
                    )
                    n_sens += 1

                    tot_extracted += n_extracted[j]

                if cutoff == n_sens:
                    break

            # clear up RAM usage for finall activation dump
            del self.model

            if create_avg_eos:
                self._normalize_avg_eos_activations(avg_eos_states, n_sens)
                self.activation_writer.dump_avg_eos(avg_eos_states)

            if not only_dump_avg_eos:
                self.activation_writer.dump_activation_ranges(activation_ranges)
                if not dynamic_dumping:
                    for name in all_activations.keys():
                        all_activations[name] = np.concatenate(
                            all_activations[name], axis=0
                        )
                    self.activation_writer.dump_activations(all_activations)

        if dynamic_dumping and not only_dump_avg_eos:
            print("\nConcatenating sequentially dumped pickle files into 1 array...")
            self.activation_writer.concat_pickle_dumps()

        print(f"\nExtraction finished.")
        print(
            f"{n_sens} sentences have been extracted, yielding {tot_extracted} data points."
        )

    def _extract_sentence(
        self, batch: Batch, examples: List[Example], selection_func: SelectFunc
    ) -> Tuple[BatchArrayDict, List[int]]:
        """ Generates the embeddings of a sentence and writes to file.

        Parameters
        ----------
        batch : Batch
            Batch containing sentence and label information.
        selection_func : SelectFunc
            Function that determines whether activations for a token
            should be extracted or not.

        Returns
        -------
        sen_activations : PartialArrayDict
            Extracted activations for this sentence. Activations are
            converted to numpy arrays.
        n_extracted : Labels
            Number of extracted activations.
        """
        batch_size = len(batch)
        n_extracted: List[int] = [0] * batch_size

        batch_activations: BatchArrayDict = self._init_batch_activations(batch_size)
        cur_activations: FullActivationDict = self.model.init_hidden(batch_size)

        sentence, sen_lens = batch.sen
        for i in range(sentence.size(1)):
            tokens = sentence[:, i]

            with torch.no_grad():
                _out, cur_activations = self.model(
                    tokens, cur_activations, compute_out=False
                )

            # Check whether current activations match criterion defined in selection_func
            for j in range(batch_size):
                if i < sen_lens[j] and selection_func(i, tokens[j], examples[j]):
                    for layer, name in self.activation_names:
                        cur_activation = cur_activations[layer][name][j]
                        if self.model.array_type == "torch":
                            cur_activation = cur_activation.detach().numpy()
                        batch_activations[j][(layer, name)].append(cur_activation)

                    n_extracted[j] += 1

        for j in range(batch_size):
            for a_name, arr in batch_activations[j].items():
                batch_activations[j][a_name] = np.array(arr)

        return batch_activations, n_extracted

    def _init_batch_activations(self, batch_size: int) -> BatchArrayDict:
        """ Initial dict of of activations for current batch. """

        return {i: self._init_sen_activations() for i in range(batch_size)}

    def _init_sen_activations(self) -> PartialArrayDict:
        """ Initial dict for each activation that is extracted. """

        return {(layer, name): [] for (layer, name) in self.activation_names}

    def _init_avg_eos_activations(self) -> FullActivationDict:
        # TODO this might break now as init_states always adds batch dim
        init_avg_eos_activations: FullActivationDict = self.init_lstm_states.create_zero_init_states()

        for layer in range(self.model.num_layers):
            if (layer, "hx") not in self.activation_names:
                self.activation_names.append((layer, "hx"))
            if (layer, "cx") not in self.activation_names:
                self.activation_names.append((layer, "cx"))

        return init_avg_eos_activations

    def _update_avg_eos_activations(
        self, prev_activations: FullActivationDict, new_activations: BatchArrayDict
    ) -> None:
        for j in new_activations.keys():
            for layer in prev_activations.keys():
                for name in prev_activations[layer].keys():
                    eos_activation = new_activations[j][(layer, name)][-1]
                    if self.model.array_type == "torch":
                        prev_activations[layer][name] += torch.from_numpy(
                            eos_activation
                        )
                    else:
                        prev_activations[layer][name] += eos_activation

    @staticmethod
    def _normalize_avg_eos_activations(
        avg_eos_activations: FullActivationDict, n_sens: int
    ) -> None:
        for layer in avg_eos_activations.keys():
            for name in avg_eos_activations[layer].keys():
                avg_eos_activations[layer][name] /= n_sens
