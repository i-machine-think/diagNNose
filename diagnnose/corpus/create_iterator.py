from torchtext.data import BucketIterator

from diagnnose.typedefs.corpus import Corpus


def create_iterator(
    corpus: Corpus, batch_size: int = 1, device: str = "cpu"
) -> BucketIterator:
    iterator = BucketIterator(
        dataset=corpus, batch_size=batch_size, device=device, shuffle=False
    )

    return iterator
