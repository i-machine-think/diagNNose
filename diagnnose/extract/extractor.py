from contextlib import ExitStack
from typing import List, Optional

import torch
from torchtext.data import Batch
from tqdm import tqdm

from diagnnose.activations import ActivationReader, ActivationWriter
from diagnnose.activations.selection_funcs import return_all
from diagnnose.corpus import Corpus
from diagnnose.corpus.create_iterator import create_iterator
from diagnnose.typedefs.activations import (
    ActivationDict,
    ActivationNames,
    ActivationRanges,
    SelectionFunc,
)

BATCH_SIZE = 1024


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
    activation_names : List[tuple[int, str]]
        List of (layer, activation_name) tuples
    activations_dir : str, optional
        Directory to which activations will be written. If not provided
        the `extract()` method will only return the activations without
        writing them to disk.
    batch_size : int, optional
        Amount of sentences processed per forward step. Higher batch
        size increases extraction speed, but should be done
        accordingly to the amount of available RAM. Defaults to 1.
    selection_func : Callable[[int, int, Example], bool]
        Function which determines if activations for a token should
        be extracted or not.
    """

    def __init__(
        self,
        model: "LanguageModel",
        corpus: Corpus,
        activation_names: ActivationNames,
        activations_dir: Optional[str] = None,
        selection_func: SelectionFunc = return_all,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.model = model
        self.corpus = corpus
        self.activation_names = activation_names
        self.selection_func = selection_func
        self.batch_size = batch_size

        self.activation_ranges = self.create_activation_ranges()

        if activations_dir is None:
            self.activation_writer: Optional[ActivationWriter] = None
        else:
            self.activation_writer = ActivationWriter(activations_dir)

    def extract(self) -> ActivationReader:
        """ Extracts embeddings from a corpus.

        Uses contextlib.ExitStack to write to multiple files at once.

        Returns
        -------
        final_activations : Union[BatchActivationTensors, ActivationDict]
            If `dynamic_dumping` is set to True, only the activations of
            the final batch are returned, in a dictionary mapping the
            batch id to the corresponding activations. If set to False
            an `ActivationDict` containing all activations is returned.
        """
        print(f"\nStarting extraction of {len(self.corpus)} sentences...")

        if self.activation_writer is not None:
            with ExitStack() as stack:
                self.activation_writer.create_output_files(stack, self.activation_names)

                self._extract_corpus(dump=True)

                self.activation_writer.dump_meta_info(
                    self.activation_ranges, self.selection_func
                )

            activation_reader = ActivationReader(
                activations_dir=self.activation_writer.activations_dir
            )
        else:
            corpus_activations = self._extract_corpus(dump=False)

            activation_reader = ActivationReader(
                activation_dict=corpus_activations,
                activation_ranges=self.activation_ranges,
                selection_func=self.selection_func,
            )

        print("Extraction finished.")

        return activation_reader

    def _extract_corpus(self, dump: bool = True) -> ActivationDict:
        tot_extracted = self.activation_ranges[-1][1]
        corpus_activations: ActivationDict = self.init_activation_dict(
            tot_extracted, dump=dump
        )

        iterator = create_iterator(
            self.corpus, batch_size=self.batch_size, device=self.model.device
        )

        n_extracted = 0

        for batch in tqdm(iterator, unit="batch"):
            n_items_in_batch = (
                self.activation_ranges[batch.sen_idx[-1]][1] - n_extracted
            )

            batch_activations = self._extract_batch(batch, n_items_in_batch)

            if dump:
                self.activation_writer.dump_activations(batch_activations)
            else:
                # Insert extracted batch activations into full corpus activations dict.
                for a_name, activations in batch_activations.items():
                    batch_slice = slice(n_extracted, n_extracted + n_items_in_batch)
                    corpus_activations[a_name][batch_slice] = batch_activations[a_name]
                n_extracted += n_items_in_batch

        return corpus_activations

    def _extract_batch(self, batch: Batch, n_items_in_batch: int) -> ActivationDict:
        sens, sen_lens = getattr(batch, self.corpus.sen_column)

        # a_name -> batch_size x max_sen_len x nhid
        compute_out = (self.model.top_layer, "out") in self.activation_names
        all_activations: ActivationDict = self.model(
            sens, sen_lens, compute_out=compute_out
        )

        # a_name -> n_items_in_batch x nhid
        batch_activations: ActivationDict = self.init_activation_dict(n_items_in_batch)
        self._select_activations(batch_activations, all_activations, batch)

        return batch_activations

    def _select_activations(
        self,
        batch_activations: ActivationDict,
        all_activations: ActivationDict,
        batch: Batch,
    ) -> None:
        sen_lens = getattr(batch, self.corpus.sen_column)[1]

        a_idx = 0

        for b_idx, sen_idx in enumerate(batch.sen_idx):
            for w_idx in range(sen_lens[b_idx].item()):
                if self.selection_func(w_idx, self.corpus[sen_idx]):
                    for a_name in batch_activations:
                        selected_activation = all_activations[a_name][b_idx][w_idx]
                        batch_activations[a_name][a_idx] = selected_activation
                    a_idx += 1

    def create_activation_ranges(self) -> ActivationRanges:
        activation_ranges: ActivationRanges = []
        tot_extracted = 0

        for item in self.corpus:
            start = tot_extracted
            sen_len = len(getattr(item, self.corpus.sen_column))
            for w_idx in range(sen_len):
                if self.selection_func(w_idx, item):
                    tot_extracted += 1

            activation_ranges.append((start, tot_extracted))

        return activation_ranges

    def init_activation_dict(self, n_items: int, dump: bool = False) -> ActivationDict:
        """ If activations are dumped we don't keep track of the full
        activation dictionary.
        """
        if dump:
            return {}

        corpus_activations = {
            a_name: torch.zeros(n_items, self.model.nhid(a_name))
            for a_name in self.activation_names
        }

        return corpus_activations
