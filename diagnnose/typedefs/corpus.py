from torchtext.data import TabularDataset

from diagnnose.vocab.create import create_vocab
from torchtext.vocab import Vocab


class Corpus(TabularDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vocab: Vocab = Vocab({}, specials=[])

    def attach_vocab(
        self, vocab_path: str, sen_column: str = "sen", notify_unk: bool = False
    ):
        attach_vocab(self, vocab_path, sen_column, notify_unk)


def attach_vocab(
    corpus: Corpus, vocab_path: str, sen_column: str = "sen", notify_unk: bool = False
) -> None:
    vocab = create_vocab(vocab_path, notify_unk=notify_unk)

    corpus.fields[sen_column].vocab = Vocab({}, specials=[])
    corpus.fields[sen_column].vocab.stoi = vocab
    corpus.fields[sen_column].vocab.itos = list(vocab.keys())

    corpus.vocab = corpus.fields[sen_column].vocab
