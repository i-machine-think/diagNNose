from typing import Any, Dict, List, Optional

from diagnnose.typedefs.corpus import Corpus, CorpusSentence


def import_corpus_from_path(corpus_path: str,
                            corpus_header: Optional[List[str]] = None,
                            to_lower: bool = False,
                            header_from_first_line: bool = False) -> Corpus:
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
    corpus_header : List[str], optional
        Optional list of attribute names of each column, if not provided
        all lines will be considered to be sentences,  with the
        attribute name "sen".
    header_from_first_line : bool, optional
        Use the first line of the corpus as the attribute names of the
        corpus.
    to_lower : bool, optional
        Transform entire corpus to lower case, defaults to False.

    Returns
    -------
    corpus : Corpus
        A Corpus type containing the parsed sentences and optional labels
    """
    corpus = {}

    init_corpus = read_raw_corpus(corpus_path,
                                  header=corpus_header,
                                  to_lower=to_lower,
                                  header_from_first_line=header_from_first_line)

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
                    header: Optional[List[str]] = None,
                    header_from_first_line: bool = False,
                    to_lower: bool = False) -> Dict[int, Dict[str, Any]]:
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
    header_from_first_line : bool, optional
        Optional toggle to use the first line of the corpus as attribute
        names for the parsed corpus, such as in a csv file.
    to_lower : bool, optional
        Transform entire corpus to lower case, defaults to False.

    Returns
    -------
    corpus : Dict[int, Dict[str, Any]]
        Dictionary mapping id to dict of attributes
    """
    with open(corpus_path) as f:
        lines = f.read().strip().split('\n')

    split_lines = [l.strip().split(separator) for l in lines]
    if header_from_first_line:
        header = split_lines[0]
        split_lines = split_lines[1:]
    elif header is None:
        header = ['sen']

    init_corpus = {
        i: string_to_dict(header, x, to_lower) for i, x in enumerate(split_lines)
    }

    return init_corpus


def string_to_dict(header: List[str], line: List[str], to_lower: bool) -> Dict[str, Any]:
    """Converts a list of attributes and values to a dictionary.

    Also splits the sentence string to a list of strings.
    """
    sendict: Dict[str, Any] = dict(zip(header, line))
    for k, v in sendict.items():
        if v.isnumeric():
            sendict[k] = int(v)

    if 'sen' in sendict:
        sen = sendict['sen'].lower() if to_lower else sendict['sen']
        sendict['sen'] = sen.strip().split(' ')
    if 'sent' in sendict:
        sen = sendict['sent'].lower() if to_lower else sendict['sent']
        sendict['sent'] = sen.strip().split(' ')

    return sendict
