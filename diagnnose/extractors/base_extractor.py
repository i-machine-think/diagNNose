from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
from torchtext.data import Batch
from tqdm import tqdm

from diagnnose.activations.activation_writer import ActivationWriter
from diagnnose.corpus.create_iterator import create_iterator
from diagnnose.typedefs.activations import (
    ActivationDict,
    ActivationListDict,
    ActivationNames,
    ActivationRanges,
    BatchActivationTensorLists,
    BatchActivationTensors,
    SelectionFunc,
)
from diagnnose.corpus import Corpus

# https://stackoverflow.com/a/39757388/3511979
if TYPE_CHECKING:
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
    activation_names : List[tuple[int, str]]
        List of (layer, activation_name) tuples
    activations_dir : str, optional
        Directory to which activations will be written. If not provided
        the `extract()` method will only return the activations without
        writing them to disk.
    """

    def __init__(
        self,
        model: LanguageModel,
        corpus: Corpus,
        activation_names: ActivationNames,
        activations_dir: Optional[str] = None,
    ) -> None:
        self.model = model
        self.corpus = corpus
        self.activation_names: ActivationNames = activation_names

        if activations_dir is not None:
            self.activation_writer = ActivationWriter(activations_dir)
        else:
            self.activation_writer = None

    def extract(
        self,
        batch_size: int = 1,
        dynamic_dumping: bool = True,
        selection_func: SelectionFunc = lambda sen_id, pos, item: True,
    ) -> Union[BatchActivationTensors, ActivationDict]:
        """ Extracts embeddings from a corpus.

        Uses contextlib.ExitStack to write to multiple files at once.
        File writing is done directly per batch if `dynamic_dumping` is
        set to True, to lessen RAM usage.

        Parameters
        ----------
        batch_size : int, optional
            Amount of sentences processed per forward step. Higher batch
            size increases extraction speed, but should be done
            accordingly to the amount of available RAM. Defaults to 1.
        dynamic_dumping : bool, optional
            Dump files dynamically, i.e. once per sentence, or dump
            all files at the end of extraction. Defaults to True.
        selection_func : Callable[[int, int, Example], bool]
            Function which determines if activations for a token should
            be extracted or not.

        Returns
        -------
        final_activations : Union[BatchActivationTensors, ActivationDict]
            If `dynamic_dumping` is set to True, only the activations of
            the final batch are returned, in a dictionary mapping the
            batch id to the corresponding activations. If set to False
            an `ActivationDict` containing all activations is returned.
        """
        tot_extracted = n_sens = 0

        dump_activations = self.activation_writer is not None

        all_activations: ActivationListDict = self._init_sen_activations()
        activation_ranges: ActivationRanges = {}
        iterator = create_iterator(
            self.corpus, batch_size=batch_size, device=self.model.device
        )

        print(f"\nStarting extraction of {len(self.corpus) } sentences...")

        with ExitStack() as stack:
            if dump_activations:
                print(
                    f"Saving activations to `{self.activation_writer.activations_dir}`"
                )
                self.activation_writer.create_output_files(stack, self.activation_names)

            for batch in tqdm(iterator, unit="batch"):
                batch_activations, n_extracted = self._extract_sentence(
                    batch, n_sens, selection_func
                )

                for j in batch_activations.keys():
                    if dump_activations and dynamic_dumping:
                        self.activation_writer.dump_activations(batch_activations[j])
                    else:
                        for name in all_activations.keys():
                            all_activations[name].append(batch_activations[j][name])

                for j in batch_activations.keys():
                    activation_ranges[n_sens + j] = (
                        tot_extracted,
                        tot_extracted + n_extracted[j],
                    )
                    tot_extracted += n_extracted[j]

                n_sens += batch.batch_size

            # clear up RAM usage for final activation dump
            del self.model

            if dynamic_dumping:
                final_activations = batch_activations
            else:
                final_activations = {}
                for name in all_activations.keys():
                    final_activations[name] = torch.cat(all_activations[name], dim=0)

            if dump_activations:
                self.activation_writer.dump_activation_ranges(activation_ranges)
                if not dynamic_dumping:
                    self.activation_writer.dump_activations(final_activations)

        if dump_activations and dynamic_dumping:
            print("\nConcatenating sequentially dumped pickle files into 1 array...")
            self.activation_writer.concat_pickle_dumps()

        print(
            "\nExtraction finished.\n"
            f"{n_sens} sentences have been extracted, yielding {tot_extracted} data points."
        )

        return final_activations

    def _extract_sentence(
        self, batch: Batch, n_sens: int, selection_func: SelectionFunc
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
        cur_activations: ActivationDict = self.model.init_hidden(batch_size)
        examples = self.corpus.examples[n_sens : n_sens + batch_size]

        sens, sen_lens = batch.sen
        for i in range(sens.size(1)):
            if self.model.use_char_embs:
                tokens = [e.sen[min(i, len(e.sen) - 1)] for e in examples]
            else:
                tokens = sens[:, i]

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

    def _init_sen_activations(self) -> ActivationListDict:
        """ Initial dict for each activation that is extracted. """

        return {(layer, name): [] for (layer, name) in self.activation_names}
