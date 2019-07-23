from typing import List, Optional

from torchtext.data import Field, RawField, TabularDataset
from torchtext.vocab import Vocab

from diagnnose.vocab import create_vocab_from_path


def import_corpus(
    corpus_path: str,
    corpus_header: Optional[List[str]] = None,
    header_from_first_line: bool = False,
    to_lower: bool = False,
    vocab_path: Optional[str] = None,
) -> TabularDataset:

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
    to_lower : bool, optional
        Transform entire corpus to lower case, defaults to False.
    header_from_first_line : bool, optional
        Use the first line of the corpus as the attribute names of the
        corpus.
    vocab_path : str, optional
        Path to the model vocabulary, which should a file containing a
        vocab entry at each line.

    Returns
    -------
    corpus : TabularDataset
        A TabularDataset containing the parsed sentences and optional labels
    """

    if corpus_header is None:
        if header_from_first_line:
            with open(corpus_path) as f:
                corpus_header = f.readline().strip().split("\t")
        else:
            corpus_header = ["sen"]

    assert "sen" in corpus_header, "`sen` should be part of corpus_header!"

    fields = {}
    for field in corpus_header:
        if field == "sen":
            fields[field] = Field(
                batch_first=True, include_lengths=True, lower=to_lower
            )
        elif field == "labels":
            fields[field] = Field(
                use_vocab=False, tokenize=lambda s: list(map(int, s.split()))
            )
        else:
            fields[field] = RawField()
            fields[field].is_target = False

    # The current torchtext Vocab does not allow a fixed vocab order
    if vocab_path is not None:
        vocab = create_vocab_from_path(vocab_path)
        fields["sen"].vocab = Vocab({}, specials=[])
        fields["sen"].vocab.stoi = vocab
        fields["sen"].vocab.itos = list(vocab.keys())

    corpus = TabularDataset(
        fields=fields.items(),
        format="tsv",
        path=corpus_path,
        skip_header=header_from_first_line,
        csv_reader_params={"quotechar": None},
    )

    if vocab_path is not None:
        corpus.vocab = corpus.fields["sen"].vocab

    return corpus
