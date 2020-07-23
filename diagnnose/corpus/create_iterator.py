from typing import Callable, Optional

from torchtext.data import Iterator

from diagnnose.corpus import Corpus


def create_iterator(
    corpus: Corpus, batch_size: int = 1, device: str = "cpu", sort: bool = False
) -> Iterator:
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
