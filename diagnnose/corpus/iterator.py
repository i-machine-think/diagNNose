from typing import Callable, Optional

from torchtext.data import Iterator

from diagnnose.corpus import Corpus


def create_iterator(
    corpus: Corpus, batch_size: int = 1, device: str = "cpu", sort: bool = False
) -> Iterator:
    """Transforms a Corpus into an :class:`torchtext.data.Iterator`.

    Parameters
    ----------
    corpus : Corpus
         Corpus containing sentences that will be tokenized and
         transformed into a batch.
    batch_size : int, optional
        Amount of sentences processed per forward step. Higher batch
        size increases processing speed, but should be done
        accordingly to the amount of available RAM. Defaults to 1.
    device : str, optional
        Torch device on which forward passes will be run.
        Defaults to cpu.
    sort : bool, optional
        Toggle to sort the corpus based on sentence length. Defaults to
        ``False``.

    Returns
    -------
    iterator : Iterator
        Iterator containing the batchified Corpus.
    """
    if sort is not None:
        sort_key: Optional[Callable] = lambda ex: len(getattr(ex, corpus.sen_column))
    else:
        sort_key = None

    iterator = Iterator(
        dataset=corpus,
        batch_size=batch_size,
        device=device,
        shuffle=False,
        sort=sort,
        sort_key=sort_key,
        train=False,
    )

    return iterator
