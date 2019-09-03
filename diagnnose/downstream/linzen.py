from typing import Any, Dict

import torch
from tqdm import tqdm

from diagnnose.corpus.create_iterator import create_iterator
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.models.lm import LanguageModel

from .misc import calc_final_hidden


# TODO: Move preproc_linzen.ipynb code to here
def linzen_init(
    vocab_path: str, path: str, device: str = "cpu", **kwargs: Any
) -> Dict[str, Any]:
    corpus = import_corpus(path, header_from_first_line=True, vocab_path=vocab_path)
    iterator = create_iterator(corpus, batch_size=2000, device=device, sort=True)

    return {"corpus": corpus, "iterator": iterator}


def linzen_downstream(
    init_dict: Dict[str, Any], model: LanguageModel, **kwargs: Any
) -> float:
    correct = 0.0
    corpus = init_dict["corpus"]
    iterator = init_dict["iterator"]

    for batch in tqdm(iterator):
        all_sens = [corpus.examples[idx].sen for idx in batch.idx]
        final_hidden = calc_final_hidden(model, batch, all_sens)

        output_classes = torch.tensor(
            [
                [
                    corpus.vocab.stoi[batch.verb[i]],
                    corpus.vocab.stoi[batch.wrong_verb[i]],
                ]
                for i in range(batch.batch_size)
            ]
        ).to(torch.long)

        probs = torch.bmm(model.decoder_w[output_classes], final_hidden.unsqueeze(2))
        probs = probs[:, :, 0]
        probs += model.decoder_b[output_classes]

        correct += int(torch.sum(probs[:, 0] > probs[:, 1]))

    accuracy = correct / len(corpus.examples)

    print(accuracy)

    return accuracy
