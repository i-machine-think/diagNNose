import os
from typing import Any, Dict, List, Optional

from rnnalyse.typedefs.corpus import Corpus, CorpusSentence
from rnnalyse.utils.paths import load_pickle


def import_corpus_from_path(corpus_path: str,
                            from_dict: bool = False,
                            header: Optional[List[str]] = None) -> Corpus:
    """ Imports a corpus from a path.

    The corpus can either be a raw string or a pickled dictionary.
    Outputs a `Corpus` type, that is used throughout the library.

    The raw sentence is assumed to be labeled `sen` or `sent`
    Sentences can possibly be labeled, which are assumed to be labeled
    by a `labels` tag.

    Parameters
    ----------
    corpus_path : str
        Path to corpus file
    from_dict : bool, optional
        Indicates whether to load a pickled dict. Defaults to False.
    header : List[str], optional
        Optional list of attribute names of each column

    Returns
    -------
    corpus : Corpus
        A Corpus type containing the parsed sentences and optional labels
    """
    corpus = {}

    if from_dict:
        init_corpus: Dict[int, Dict[str, Any]] = load_pickle(os.path.expanduser(corpus_path))
    else:
        init_corpus = read_raw_corpus(corpus_path, header=header)

    for key, item in init_corpus.items():
        assert 'sen' in item.keys() or 'sent' in item.keys(), \
            'Corpus item should contain a sentence (\'sen\' or \'sent\') attribute'

        sen = item['sent'] if 'sent' in item else item['sen']
        labels = item['labels'] if 'labels' in item else None
        misc_info = {k: v for k, v in item.items() if k not in ['sen', 'sent', 'labels']}

        corpus_sentence = CorpusSentence(sen, labels, misc_info)
        corpus_sentence.validate()
        corpus[key] = corpus_sentence

    return corpus


def read_raw_corpus(corpus_path: str,
                    separator: str = '\t',
                    header: Optional[List[str]] = None) -> Dict[int, Dict[str, Any]]:
    """ Reads a tsv/csv type file and converts it to a dictionary.

    Expects the first line to indicate the column names if header is not
    provided.

    Parameters
    ----------
    corpus_path : str
        Path to corpus file
    separator : str, optional
        Character separator of each attribute in the corpus. Defaults to \t
    header : List[str], optional
        Optional list of attribute names of each column
    """
    with open(corpus_path) as f:
        lines = f.read().strip().split('\n')

    split_lines = [l.strip().split(separator) for l in lines]
    if header is None:
        header = split_lines[0]
    corpus_lines = split_lines[1:]

    init_corpus = {
        i: string_to_dict(header, x) for i, x in enumerate(corpus_lines)
    }

    return init_corpus


def string_to_dict(header: List[str], line: List[str]) -> Dict[str, Any]:
    """Converts a list of attributes and values to a dictionary.

    Also splits the sentence string to a list of strings.
    """
    sendict: Dict[str, Any] = dict(zip(header, line))
    if 'sen' in sendict:
        sendict['sen'] = sendict['sen'].strip().split(' ')
    if 'sent' in sendict:
        sendict['sent'] = sendict['sent'].strip().split(' ')

    return sendict
