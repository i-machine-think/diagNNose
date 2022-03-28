from contextlib import ExitStack
from typing import TYPE_CHECKING, Optional, Union

import torch
from torchtext.data import Batch
from tqdm import tqdm

import diagnnose.activations.selection_funcs as selection_funcs
from diagnnose.activations import ActivationReader, ActivationWriter
from diagnnose.activations.selection_funcs import return_all
from diagnnose.corpus import Corpus, create_iterator
from diagnnose.typedefs.activations import (
    ActivationDict,
    ActivationNames,
    ActivationRanges,
    SelectionFunc,
)

if TYPE_CHECKING:
    from diagnnose.models import LanguageModel

BATCH_SIZE = 1024


class Extractor:
    """Extracts all intermediate activations of a LM from a corpus.

    Only activations that are provided in activation_names will be
    stored in a pickle file. Each activation is written to its own file.

    Parameters
    ----------
    model : LanguageModel
        Language model that inherits from LanguageModel.
    corpus : Corpus
        Corpus containing sentences to be extracted.
    activation_names : List[tuple[int, str]], optional
        List of (layer, activation_name) tuples. If not provided all
        activation_names corresponding to the ``model`` will be
        extracted.
    activations_dir : str, optional
        Directory to which activations will be written. If not provided
        the `extract()` method will only return the activations without
        writing them to disk.
    selection_func : Union[SelectionFunc, str]
        Function which determines if activations for a token should
        be extracted or not. Can also be provided as a string,
        indicating the method name of one of the default
        selection_funcs in
        :py:mod:`diagnnose.activations.selection_funcs`.
    batch_size : int, optional
        Amount of sentences processed per forward step. Higher batch
        size increases extraction speed, but should be done
        accordingly to the amount of available RAM. Defaults to 1.
    """

    def __init__(
        self,
        model: "LanguageModel",
        corpus: Corpus,
        activation_names: Optional[ActivationNames] = None,
        activations_dir: Optional[str] = None,
        selection_func: Union[SelectionFunc, str] = return_all,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.model = model
        self.corpus = corpus
        self.activation_names = activation_names or model.activation_names()
        if isinstance(selection_func, str):
            self.selection_func: SelectionFunc = getattr(
                selection_funcs, selection_func
            )
        else:
            self.selection_func: SelectionFunc = selection_func
        self.batch_size = batch_size

        self.activation_ranges = []
        self.set_activation_ranges()

        if activations_dir is None:
            self.activation_writer: Optional[ActivationWriter] = None
        else:
            self.activation_writer = ActivationWriter(activations_dir)

    def extract(self) -> ActivationReader:
        """Extracts embeddings from a corpus.

        Uses :class:`contextlib.ExitStack` to write to multiple files
        simultaneously.

        Returns
        -------
        activation_reader : ActivationReader
            After extraction an activation_reader is returned that
            provides direct access to the extracted activations.
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
                activations_dir=self.activation_writer.activations_dir,
                activation_names=self.activation_names,
            )
        else:
            corpus_activations = self._extract_corpus(dump=False)

            activation_reader = ActivationReader(
                activation_dict=corpus_activations,
                activation_names=self.activation_names,
                activation_ranges=self.activation_ranges,
                selection_func=self.selection_func,
            )

        n_extracted = self.activation_ranges[-1][-1]
        print(f"Extraction finished, {n_extracted} activations have been extracted.")

        return activation_reader

    def _extract_corpus(self, dump: bool = True) -> ActivationDict:
        tot_extracted = self.activation_ranges[-1][1]
        corpus_activations: ActivationDict = self._init_activation_dict(
            tot_extracted, dump=dump
        )

        corpus = self._filter_corpus()

        iterator = create_iterator(
            corpus, batch_size=self.batch_size, device=self.model.device
        )

        for batch in tqdm(iterator, unit="batch"):
            batch_activations = self._extract_batch(batch)

            if dump:
                self.activation_writer.dump_activations(batch_activations)
            else:
                # Insert extracted batch activations into full corpus activations dict.
                batch_start = self.activation_ranges[batch.sen_idx[0]][0]
                batch_stop = self.activation_ranges[batch.sen_idx[-1]][1]
                for a_name, activations in batch_activations.items():
                    corpus_activations[a_name][batch_start:batch_stop] = activations

        return corpus_activations

    def _filter_corpus(self) -> Corpus:
        """ Skip items for which selection_func yields 0 activations. """
        sen_ids = [
            idx
            for idx, (start, stop) in enumerate(self.activation_ranges)
            if start != stop
        ]

        if len(sen_ids) != len(self.corpus):
            return self.corpus.slice(sen_ids)

        return self.corpus

    def _extract_batch(self, batch: Batch) -> ActivationDict:
        """Processes the items in `batch` and selects the activations
        that should should be extracted according to selection_func.
        """
        sens, sen_lens = getattr(batch, self.corpus.sen_column)

        compute_out = any("out" in a_name for a_name in self.activation_names)
        kwargs = {}
        if getattr(self.model, "compute_pseudo_ll", False):
            kwargs["mask_idx"] = self.corpus.tokenizer.mask_token_id
            kwargs["selection_func"] = self.selection_func
            kwargs["batch"] = batch

        with torch.no_grad():
            # a_name -> batch_size x max_sen_len x nhid
            all_activations: ActivationDict = self.model(
                input_ids=sens,
                input_lengths=sen_lens,
                compute_out=compute_out,
                only_return_top_embs=False,
                **kwargs,
            )

        # a_name -> n_items_in_batch x nhid
        batch_activations = self._select_activations(all_activations, batch)

        return batch_activations

    def _select_activations(
        self,
        all_activations: ActivationDict,
        batch: Batch,
    ) -> ActivationDict:
        """ Selects only the activations that pass selection_func. """
        batch_start = self.activation_ranges[batch.sen_idx[0]][0]
        batch_end = self.activation_ranges[batch.sen_idx[-1]][1]
        n_items_in_batch = batch_end - batch_start
        batch_activations: ActivationDict = self._init_activation_dict(n_items_in_batch)

        a_idx = 0

        for b_idx, sen_idx in enumerate(batch.sen_idx):
            item = self.corpus[sen_idx]
            sen_len = len(getattr(item, self.corpus.sen_column))
            for w_idx in range(sen_len):
                if self.selection_func(w_idx, item):
                    for a_name in batch_activations:
                        selected_activation = all_activations[a_name][b_idx][w_idx]
                        batch_activations[a_name][a_idx] = selected_activation
                    a_idx += 1

        return batch_activations

    def set_activation_ranges(self) -> None:
        activation_ranges: ActivationRanges = []
        tot_extracted = 0

        for item in self.corpus:
            start = tot_extracted
            sen_len = len(getattr(item, self.corpus.sen_column))
            for w_idx in range(sen_len):
                if self.selection_func(w_idx, item):
                    tot_extracted += 1

            activation_ranges.append((start, tot_extracted))

        self.activation_ranges = activation_ranges

    def _init_activation_dict(self, n_items: int, dump: bool = False) -> ActivationDict:
        # If activations are dumped we don't keep track of the full activation dictionary,
        # so an empty dictionary is returned.
        if dump:
            return {}

        corpus_activations = {
            a_name: torch.zeros(
                n_items, self.model.nhid(a_name), device=self.model.device
            )
            for a_name in self.activation_names
        }

        return corpus_activations
