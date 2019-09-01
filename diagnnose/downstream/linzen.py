from typing import Any, Dict

import torch
from torch.nn.utils.rnn import pack_padded_sequence
from tqdm import tqdm

from diagnnose.corpus.create_iterator import create_iterator
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.models.lm import LanguageModel
from diagnnose.typedefs.activations import DTYPE


def linzen_init(
    vocab_path: str, path: str, device: str = "cpu", **kwargs: Any
) -> Dict[str, Any]:
    corpus = import_corpus(path, header_from_first_line=True, vocab_path=vocab_path)
    iterator = create_iterator(corpus, batch_size=2000, device=device, sort=True)

    return {"corpus": corpus, "iterator": iterator}


def linzen_downstream(init_dict: Dict[str, Any], model: LanguageModel) -> float:
    correct = 0.0
    corpus = init_dict["corpus"]
    iterator = init_dict["iterator"]

    for batch in tqdm(iterator):
        sens, slens = batch.sen
        batch_size = batch.batch_size
        packed_sens = pack_padded_sequence(sens, lengths=slens, batch_first=True)

        hidden = model.init_hidden(batch_size)
        final_hidden = torch.zeros((batch_size, model.output_size), dtype=DTYPE).to(
            sens.device
        )
        n = 0
        for i, j in enumerate(packed_sens.batch_sizes):
            w = packed_sens[0][n : n + j]
            for name, v in hidden.items():
                hidden[name] = v[:j]
            if hasattr(model, "use_char_embs") and model.use_char_embs:
                w = [corpus.examples[batch.idx[k]].sen[i] for k in range(j)]
            _, hidden = model(w, hidden, compute_out=False)
            for k in range(int(j)):
                final_hidden[k] = hidden[model.top_layer, "hx"][k]
            n += j

        output_classes = torch.tensor(
            [
                [
                    corpus.vocab.stoi[batch.verb[i]],
                    corpus.vocab.stoi[batch.wrong_verb[i]],
                ]
                for i in range(batch_size)
            ]
        ).to(torch.long)
        probs = torch.bmm(model.decoder_w[output_classes], final_hidden.unsqueeze(2))
        probs = probs[:, :, 0]
        probs += model.decoder_b[output_classes]

        correct += int(torch.sum(probs[:, 0] > probs[:, 1]))

    accuracy = correct / len(corpus.examples)

    print(accuracy)

    return accuracy
