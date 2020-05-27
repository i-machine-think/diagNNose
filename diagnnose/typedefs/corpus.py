from torchtext.data import TabularDataset

from diagnnose.vocab.create import create_vocab
from torchtext.vocab import Vocab


class Corpus(TabularDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.vocab: Vocab = Vocab({}, specials=[])

    def attach_vocab(
        self, vocab_path: str, sen_column: str = "sen", notify_unk: bool = False
    ) -> None:
        vocab = create_vocab(vocab_path, notify_unk=notify_unk)

        self.fields[sen_column].vocab = Vocab({}, specials=[])
        self.fields[sen_column].vocab.stoi = vocab
        self.fields[sen_column].vocab.itos = list(vocab.keys())

        self.vocab = self.fields[sen_column].vocab
