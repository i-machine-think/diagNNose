""" Adapter method that preserves legacy corpus import method. """
from typing import List, Optional

from .corpus import Corpus


def import_corpus(
    path: Optional[str] = None,
    header: Optional[List[str]] = None,
    header_from_first_line: bool = False,
    to_lower: bool = False,
    vocab_path: Optional[str] = None,
    sen_column: str = "sen",
    labels_column: str = "labels",
    tokenize_columns: Optional[List[str]] = None,
    sep: str = "\t",
    create_pos_tags: bool = False,
    notify_unk: bool = False,
) -> Corpus:
    """ Imports a corpus from a path.

    The corpus can either be a raw string or a pickled dictionary.
    Outputs a `Corpus` type, that is used throughout the library.

    The raw sentence is assumed to be labeled `sen`.
    Sentences can optionally be labeled, which are assumed to be labeled
    by a `labels` tag.

    Parameters
    ----------
    path : str
        Path to corpus file.
    header : List[str], optional
        Optional list of attribute names of each column. If not provided
        all lines will be considered to be sentences, with the
        attribute name "sen". In case the corpus file contains 2 columns
        the header ["sen", "labels"] will be assumed.
    header_from_first_line : bool, optional
        Use the first line of the corpus as the attribute names of the
        corpus.
    to_lower : bool, optional
        Transform entire corpus to lower case, defaults to False.
    vocab_path : str, optional
        Path to the model vocabulary, which should a file containing a
        vocab entry at each line.
    sen_column : str, optional
        Name of the corpus column containing the raw sentences.
        Defaults to `sen`.
    labels_column : str, optional
        Name of the corpus column containing the sentence labels.
        Defaults to `labels`.
    tokenize_columns : List[str], optional
        List of column names that should be tokenized according to the
        provided vocabulary.
    sep : str, optional
        Column separator of corpus file, either a tsv or csv.
        Defaults to '\t'.
    create_pos_tags : bool, optional
        Toggle to create POS tags for each item. Defaults to False.
    notify_unk : bool, optional
        Notify when a requested token is not present in the vocab.
        Defaults to False.

    Returns
    -------
    corpus : Corpus
        A Corpus containing the parsed sentences and optional labels
    """

    corpus = Corpus.create(
        path,
        header=header,
        header_from_first_line=header_from_first_line,
        to_lower=to_lower,
        sen_column=sen_column,
        labels_column=labels_column,
        sep=sep,
        vocab_path=vocab_path,
        notify_unk=notify_unk,
        tokenize_columns=tokenize_columns,
        create_pos_tags=create_pos_tags,
    )

    return corpus
