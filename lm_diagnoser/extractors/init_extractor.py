from typing import BinaryIO, Dict, List, Optional, Tuple

import torch
import pickle
from time import time
from contextlib import ExitStack
from models.language_model import LanguageModel
from corpus.labeled import LabeledCorpus, LabeledSentence
from embeddings.initial import InitEmbs


ActivationName = Tuple[int, str]
ActivationFiles = Dict[ActivationName, BinaryIO]


class Extractor:
    """ Extracts all intermediate activations of a LM from a corpus.

    Only activations that are provided in activation_names will be
    stored in a pickle file. Each activation is written to its own file.

    Parameters
    ----------
    model : LanguageModel
        Language model that inherits from LanguageModel.
    corpus_path : str
        Location of labeled corpus.
    activation_names : List[ActivationName]
        List of activations to be stored of the form [(l, n)] with
            l (int): the layer number.
            n (str): activation name as defined in model.
    init_embs_path : str, optional (default = '')
        Location of initial embeddings.

    Attributes
    ----------
    model : LanguageModel
        Language model that inherits from LanguageModel.
    hidden_size : int
        Number of hidden units in model.
    activation_names : List[ActivationName]
        List of activations to be stored.
    activation_files : ActivationFiles
        Dict of files to which activations will be written.
    label_file: Optional[BinaryIO]
        File to which sentence labels will be written.
    corpus : LabeledCorpus
        corpus containing the labels for each sentence.
    num_extracted : int
        Total number of extracted activations (1 per token)
    """
    def __init__(self,
                 model: LanguageModel,
                 corpus_path: str,
                 activation_names: List[ActivationName],
                 init_embs_path: str = '') -> None:

        self.model = model
        self.hidden_size = model.hidden_size
        self.activation_names = activation_names
        self.activation_files: ActivationFiles = {}
        self.label_file: Optional[BinaryIO] = None
        self.num_extracted: int = 0

        with open(corpus_path, 'rb') as f:
            self.corpus: LabeledCorpus = pickle.load(f)

        self.init_embs = InitEmbs(init_embs_path, model).activations

    # TODO: Allow batch input
    def extract(self,
                output_path: str,
                cutoff: int = -1,
                print_every: int = 100) -> None:
        """ Extracts embeddings from a labeled corpus.

        Uses an ExitStack to write to multiple files at once.

        Parameters
        ----------
        output_path : str
            Directory to which extractions will be saved.
        cutoff : int, optional (default = -1)
            Number of sentences to be extracted. Set to -1 to extract
            entire corpus.
        print_every : int, optional (default = 100)
            Print progress info every n steps.
        """
        start_time: float = time()

        with ExitStack() as stack:
            self.create_output_files(output_path, stack)

            for n, labeled_sentence in enumerate(self.corpus):
                if n % print_every == 0:
                    print(f'{n}\t{time() - start_time:.2f}s')

                self.extract_sentence(labeled_sentence)

                self.num_extracted += len(labeled_sentence.sen)

                if cutoff == n:
                    break

        n_sens = len(self.corpus) if cutoff == -1 else cutoff
        print(f'\nExtraction finished.')
        print(f'{n_sens} sentences have been extracted, yielding {self.num_extracted} data points.')
        print(f'Total time took {time() - start_time:.2f}s')

    def create_output_files(self, output_path: str, stack: ExitStack) -> None:
        self.activation_files = {
            (l, name):
                stack.enter_context(
                    open(f'{output_path}/{name}_l{l}.pickle', 'wb')
                )
            for (l, name) in self.activation_names
        }
        self.label_file = stack.enter_context(
            open(f'{output_path}/labels.pickle', 'wb')
        )

    def extract_sentence(self, labeled_sentence: LabeledSentence) -> None:
        labeled_sentence.validate()

        sen_activations = {
            (l, name): torch.zeros(len(labeled_sentence), self.hidden_size)
            for (l, name) in self.activation_names
        }

        activations = self.init_embs

        for i, token in enumerate(labeled_sentence.sen):
            out, activations = self.model(token, activations)

            for l, name in self.activation_names:
                sen_activations[(l, name)][i] = activations[l][name]

        for l, name in self.activation_names:
            pickle.dump(sen_activations[(l, name)], self.activation_files[(l, name)])

        pickle.dump(sentence.labels, self.label_file)
