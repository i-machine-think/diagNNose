from typing import BinaryIO, Dict, List, Type, Tuple

import torch
import pickle
from time import time
from contextlib import ExitStack
from pathlib import Path
from models.language_model import ActivationDict, LanguageModel


ActivationName = Tuple[int, str]
ActivationNames = List[ActivationName]
ActivationFiles = Dict[ActivationName, BinaryIO]
Sentence = List[str]


class Extractor:
    """ Extracts all intermediate activations of a LM from a corpus.

    Only activations that are provided in extraction_names will be
    stored in a pickle file. Each activation is written to its own file.

    Args:
        model: Language model that inherits from AbstractLM
        corpus_path: Location of labeled corpus.
        extraction_names: List of activations to be extracted of the form
            [(l, n)] with
                l (int): the layer number.
                n (str): activation name as defined in model.
        init_embs_path (optional): Location of initial embeddings.
    """
    def __init__(self,
                 model: Type[LanguageModel],
                 corpus_path: str,
                 extraction_names: ActivationNames,
                 init_embs_path: str = '') -> None:
        self.model = model
        self.hidden_size = model.hidden_size
        self.extraction_names = extraction_names

        with open(corpus_path, 'rb') as f:
            self.corpus = pickle.load(f)

        self.init_embs = self.create_init_embs(init_embs_path)

    def create_init_embs(self, init_embs_path: str) -> ActivationDict:
        """
        Create initial embeddings from path or create zero initialized
        embeddings.

        :param init_embs_path: location of init embeddings
        :return: init_embs: ActivationDict with init embeddings
        """
        if init_embs_path:
            assert Path(init_embs_path).is_file(), 'File does not exist'

            with open(init_embs_path, 'rb') as f:
                init_embs = pickle.load(f)

            assert len(init_embs) == self.model.num_layers, \
                'Number of initial layers not correct'
            assert all(
                    'hx' in a.keys() and 'cx' in a.keys()
                    for a in init_embs.values()
                ), 'Initial layer names not correct, should be hx and cx'
            assert len(init_embs[0]['hx']) == self.hidden_size, \
                'Initial activation size is incorrect'

            return init_embs

        return {
            l: {
                'hx': torch.zeros(self.hidden_size),
                'cx': torch.zeros(self.hidden_size)
            } for l in range(self.model.num_layers)
        }

    def extract(self,
                output_path: str,
                cutoff: int = 32,
                print_every: int = 10) -> None:

        start_time: float = time()

        with ExitStack() as stack:
            files: ActivationFiles = {
                (l, name):
                    stack.enter_context(
                        open(f'{output_path}/{name}_l{l}.pickle', 'wb')
                    )
                for (l, name) in self.extraction_names
            }
            labels: BinaryIO = stack.enter_context(
                open(f'{output_path}/labels.pickle', 'wb')
            )

            tot_n = self.extract_corpus(files, labels, start_time, cutoff, print_every)

        n_sens = len(self.corpus) if cutoff == -1 else cutoff
        print(f'\nExtraction finished.')
        print(f'{n_sens} sentences have been extracted, yielding {tot_n} data points.')
        print(f'Total time took {time() - start_time:.2f}s')

    def extract_corpus(self,
                       files: ActivationFiles,
                       labels: BinaryIO,
                       start_time: float,
                       cutoff: int,
                       print_every: int) -> int:

        tot_n = 0

        for n, sentence in enumerate(self.corpus.values()):
            if n % print_every == 0:
                print(f'{n}\t{time() - start_time:.2f}s')

            self.extract_sentence(sentence, files, labels)

            tot_n += len(sentence.sen)

            if cutoff == n:
                break

        return tot_n

    def extract_sentence(self,
                         sentence: Sentence,
                         files: ActivationFiles,
                         labels: BinaryIO) -> None:

        sen_activations = {
            (l, name): torch.zeros(len(sentence.sen), self.hidden_size)
            for (l, name) in self.extraction_names
        }

        activations = self.init_embs

        for i, token in enumerate(sentence.sen):
            out, activations = self.model(token, activations)

            for l, name in self.extraction_names:
                sen_activations[(l, name)][i] = activations[l][name]

        for l, name in self.extraction_names:
            pickle.dump(sen_activations[(l, name)], files[(l, name)])

        pickle.dump(sentence.labels, labels)
