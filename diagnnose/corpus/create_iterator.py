from typing import Callable, Optional

from torchtext.data import BucketIterator

from diagnnose.typedefs.corpus import Corpus


def create_iterator(
    corpus: Corpus,
    batch_size: int = 1,
    device: str = "cpu",
    sort: Optional[bool] = None,
) -> BucketIterator:
    if sort is not None:
        sort_key: Optional[Callable] = lambda e: len(e.sen)
    else:
        sort_key = None

    iterator = BucketIterator(
        dataset=corpus,
        batch_size=batch_size,
        device=device,
        shuffle=False,
        sort=sort,
        sort_key=sort_key,
    )

    return iterator
