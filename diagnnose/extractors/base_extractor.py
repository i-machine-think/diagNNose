from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
from torchtext.data import Batch
from tqdm import tqdm

from diagnnose.activations.activation_writer import ActivationWriter
from diagnnose.corpus.create_iterator import create_iterator
from diagnnose.typedefs.activations import (
    ActivationNames,
    ActivationRanges,
    ActivationTensorLists,
    ActivationTensors,
    BatchActivationTensorLists,
    BatchActivationTensors,
    SelectFunc,
)
from diagnnose.typedefs.corpus import Corpus

# https://stackoverflow.com/a/39757388/3511979
if TYPE_CHECKING:
    from diagnnose.models.lm import LanguageModel


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
    activations_dir : str, optional
        Directory to which activations will be written. Should always
        be provided unless `only_return_avg_eos` is set to True in
        `self.extract`.
    activation_names : List[tuple[int, str]]
        List of (layer, activation_name) tuples

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
        self,
        model: LanguageModel,
        corpus: Corpus,
        activations_dir: Optional[str] = None,
        activation_names: Optional[ActivationNames] = None,
    ) -> None:
        self.model = model
        self.corpus = corpus

        self.activation_names: ActivationNames = activation_names or []

        if activations_dir is not None:
            self.activation_writer = ActivationWriter(activations_dir)

    # TODO: refactor
    def extract(
        self,
        batch_size: int = 1,
        cutoff: int = -1,
        dynamic_dumping: bool = True,
        selection_func: SelectFunc = lambda sen_id, pos, item: True,
        create_avg_eos: bool = False,
        only_return_avg_eos: bool = False,
    ) -> Optional[ActivationTensors]:
        """ Extracts embeddings from a corpus.

        Uses contextlib.ExitStack to write to multiple files at once.
        File writing is done directly per sentence, to lessen RAM usage.

        Parameters
        ----------
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
            Toggle to save average end of sentence activations. If set
            to True other activations won't be dumped.
        only_return_avg_eos : bool, optional
            Toggle to not dump the avg eos activations.
        """
        tot_extracted = n_sens = 0

        dump_activations = not create_avg_eos
        dump_avg_eos = create_avg_eos and not only_return_avg_eos
        if dump_activations or dump_avg_eos:
            assert hasattr(
                self, "activation_writer"
            ), "Attempting to dump activations but no activation_dir has been provided"

        all_activations: ActivationTensorLists = self._init_sen_activations()
        activation_ranges: ActivationRanges = {}
        iterator = create_iterator(
            self.corpus, batch_size=batch_size, device=self.model.device
        )

        tot_num = len(self.corpus) if cutoff == -1 else cutoff
        print(f"\nStarting extraction of {tot_num} sentences...")

        with ExitStack() as stack:
            if dump_activations or dump_avg_eos:
                self.activation_writer.create_output_files(
                    stack,
                    self.activation_names,
                    dump_activations=dump_activations,
                    dump_avg_eos=dump_avg_eos,
                )

            if create_avg_eos:
                avg_eos_states = self._init_avg_eos_activations()

            for batch in tqdm(iterator, unit="batch"):
                batch_activations, n_extracted = self._extract_sentence(
                    batch, n_sens, selection_func
                )

                if dump_activations:
                    for j in batch_activations.keys():
                        if dynamic_dumping:
                            self.activation_writer.dump_activations(
                                batch_activations[j]
                            )
                        else:
                            for name in all_activations.keys():
                                all_activations[name].append(batch_activations[j][name])

                if create_avg_eos:
                    self._update_avg_eos_activations(avg_eos_states, batch_activations)

                for j in batch_activations.keys():
                    activation_ranges[n_sens + j] = (
                        tot_extracted,
                        tot_extracted + n_extracted[j],
                    )
                    tot_extracted += n_extracted[j]

                n_sens += batch.batch_size
                if 0 < cutoff <= n_sens:
                    break

            # clear up RAM usage for final activation dump
            del self.model

            if create_avg_eos:
                self._normalize_avg_eos_activations(avg_eos_states, n_sens)
                if dump_avg_eos:
                    self.activation_writer.dump_avg_eos(avg_eos_states)
                return avg_eos_states

            if dump_activations:
                self.activation_writer.dump_activation_ranges(activation_ranges)
                if not dynamic_dumping:
                    concat_activations: ActivationTensors = {}
                    for name in all_activations.keys():
                        concat_activations[name] = torch.cat(
                            all_activations[name], dim=0
                        )
                    self.activation_writer.dump_activations(concat_activations)

        if dynamic_dumping and dump_activations:
            print("\nConcatenating sequentially dumped pickle files into 1 array...")
            self.activation_writer.concat_pickle_dumps()

        print(f"\nExtraction finished.")
        print(
            f"{n_sens} sentences have been extracted, yielding {tot_extracted} data points."
        )

        return None

    def _extract_sentence(
        self, batch: Batch, n_sens: int, selection_func: SelectFunc
    ) -> Tuple[BatchActivationTensors, List[int]]:
        """ Generates the embeddings of a sentence and writes to file.

        Parameters
        ----------
        batch : Batch
            Batch containing sentence and label information.
        n_sens : int
            Number of sentences extracted so far. Used for indexing the
            items in the batch.
        selection_func : SelectFunc
            Function that determines whether activations for a token
            should be extracted or not.

        Returns
        -------
        sen_activations : BatchTensorDict
            Dict mapping batch id's to activation names to tensors.
        n_extracted : List[int]
            Number of extracted activations, per batch item.
        """
        batch_size = len(batch)
        n_extracted: List[int] = [0] * batch_size

        batch_tensor_list: BatchActivationTensorLists = self._init_batch_activations(
            batch_size
        )
        cur_activations: ActivationTensors = self.model.init_hidden(batch_size)
        examples = self.corpus.examples[n_sens : n_sens + batch_size]

        sentence, sen_lens = batch.sen
        for i in range(sentence.size(1)):
            if self.model.use_char_embs:
                tokens = [e.sen[i] for e in examples]  # TODO: fix for uneven sen lens
            else:
                tokens = sentence[:, i]

            with torch.no_grad():
                _out, cur_activations = self.model(
                    tokens, cur_activations, compute_out=False
                )

            # Check whether current activations match criterion defined in selection_func
            for j in range(batch_size):
                if i < sen_lens[j] and selection_func(n_sens + j, i, examples[j]):
                    for layer, name in self.activation_names:
                        cur_activation = cur_activations[layer, name][j]

                        batch_tensor_list[j][(layer, name)].append(cur_activation)

                    n_extracted[j] += 1

        batch_tensors: BatchActivationTensors = {}
        for j in range(batch_size):
            batch_tensors[j] = {}
            for a_name, tensor_list in batch_tensor_list[j].items():
                if len(tensor_list) > 0:
                    batch_tensors[j][a_name] = torch.stack(tensor_list)
                else:
                    del batch_tensors[j]
                    break

        return batch_tensors, n_extracted

    def _init_batch_activations(self, batch_size: int) -> BatchActivationTensorLists:
        """ Initial dict of of activations for current batch. """

        return {i: self._init_sen_activations() for i in range(batch_size)}

    def _init_sen_activations(self) -> ActivationTensorLists:
        """ Initial dict for each activation that is extracted. """

        return {(layer, name): [] for (layer, name) in self.activation_names}

    def _init_avg_eos_activations(self) -> ActivationTensors:
        init_avg_eos_activations: ActivationTensors = self.model.create_zero_state()

        for layer in range(self.model.num_layers):
            if (layer, "hx") not in self.activation_names:
                self.activation_names.append((layer, "hx"))
            if (layer, "cx") not in self.activation_names:
                self.activation_names.append((layer, "cx"))

        return init_avg_eos_activations

    @staticmethod
    def _update_avg_eos_activations(
        prev_activations: ActivationTensors, new_activations: BatchActivationTensors
    ) -> None:
        for j in new_activations.keys():
            for layer, name in prev_activations.keys():
                eos_activation = new_activations[j][layer, name][-1]
                prev_activations[layer, name] += eos_activation

    @staticmethod
    def _normalize_avg_eos_activations(
        avg_eos_activations: ActivationTensors, n_sens: int
    ) -> None:
        if n_sens == 0:
            return
        for layer, name in avg_eos_activations.keys():
            avg_eos_activations[layer, name] = avg_eos_activations[layer, name] / n_sens
