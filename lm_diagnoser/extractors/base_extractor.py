import pickle
from contextlib import ExitStack
from time import time
from typing import Any, BinaryIO, Dict, List, Optional
import json

import torch

import numpy as np

from corpus.import_corpus import convert_to_labeled_corpus
from embeddings.initial import InitEmbs
from models.import_model import import_model_from_json
from models.language_model import LanguageModel
from typedefs.corpus import LabeledCorpus, Labels, Sentence
from typedefs.models import (
    ActivationFiles, ActivationName, FullActivationDict, PartialActivationDict)

OUTPUT_EMBS_DIR = './embeddings/data/extracted'


class Extractor:
    """ Extracts all intermediate activations of a LM from a corpus.

    Only activations that are provided in activation_names will be
    stored in a pickle file. Each activation is written to its own file.

    Arguments
    ----------
    model : str
        Path to the model to extract activations from
    vocab: str
        Path to the vocabulary of the model
    corpus : str
        Path to a pickled labeled corpus to extract 
        activations for
    activation_names : List[tuple[str, int]]
        List of (activation_name, layer) tuples
    output_dir: str, optional
        Path to output directory to write activations to
    init_embs: str, optional
        Path to pickled initial embeddings
    print_every: int, optional
        How often to print progress
    cutoff: int, optional
        How many sentences of the corpus to extract activations for

    Attributes
    ----------
    model : LanguageModel
        Language model that inherits from LanguageModel.
    corpus : LabeledCorpus
        corpus containing the labels for each sentence.
    hidden_size : int
        Number of hidden units in model.
    activation_names : List[ActivationName]
        List of activations to be stored.
    activation_files : ActivationFiles
        Dict of files to which activations will be written.
    label_file: Optional[BinaryIO]
        File to which sentence labels will be written.
    init_embs : FullActivationDict
        initial embeddings that are loaded from file or set to zero.
    """
    def __init__(self, model: str, vocab: str, corpus: LabeledCorpus,
            load_modules: str = '', 
            activation_names: List[ActivationName] = [('hx', 1), ('cx', 1)],
            output_dir: str = '',
            init_embs: str = '',
            print_every: int = 20,
            cutoff: int = -1
            ) -> None:
        self.config = locals()
        # with open(config_location) as f:
        #     self.config: Dict[str, Any] = json.load(f)
        self._validate_config(self.config)

        self.model: LanguageModel = import_model_from_json(model, vocab, 
                load_modules)
        self.corpus: LabeledCorpus = convert_to_labeled_corpus(corpus)

        self.hidden_size: int = self.model.hidden_size
        self.activation_names: List[ActivationName] = activation_names

        self.activation_files: ActivationFiles = {}
        self.label_file: Optional[BinaryIO] = None
        self.keys_file: Optional[BinaryIO] = None

        self.init_embs: InitEmbs = InitEmbs(init_embs, self.model)

    def _validate_config(self, config: Dict[str, Any]) -> None:
        pass

    # TODO: Allow batch input
    def extract(self) -> None:
        """ Extracts embeddings from a labeled corpus.

        Uses contextlib.ExitStack to write to multiple files at once.
        File writing is done directly per sentence, to lessen RAM usage.

        Extraction configurations come from `self.config`.
        Setting `cutoff` to -1 will extract the entire corpus, otherwise
        extraction is halted after extracting n sentences.
        """
        start_time: float = time()

        print_every = self.config.get('print_every', 10)
        cutoff = self.config.get('cutoff', -1)

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
        output_path = self.config.get('output_dir', OUTPUT_EMBS_DIR)

        self.activation_files = {
            (l, name):
                stack.enter_context(
                    open(f'{output_path}/{name}_l{l}.pickle', 'wb')
                )
            for (l, name) in self.activation_names
        }
        self.keys_file = stack.enter_context(
            open(f'{output_path}/keys.pickle', 'wb')
        )
        self.label_file = stack.enter_context(
            open(f'{output_path}/labels.pickle', 'wb')
        )

    def _extract_sentence(self, sentence: Sentence) -> None:
        """ Generates the embeddings of a sentence and writes to file.

        Parameters
        ----------
        sentence : Sentence
            The to-be-extracted sentence, represented as a list of strings.
        """
        sen_activations: PartialActivationDict = self._init_sen_activations(len(sentence))

        activations: FullActivationDict = self.init_embs.activations

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
            (l, name): np.empty((sen_len, self.hidden_size))
            for (l, name) in self.activation_names
        }
