import warnings
from typing import List, Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence
from torchtext.data import Batch
from torchtext.vocab import Vocab

import diagnnose.typedefs.config as config
from diagnnose.models.lm import LanguageModel


def calc_final_hidden(
    model: LanguageModel,
    batch: Batch,
    all_sens: List[List[str]],
    sen_column: str = "sen",
    skip_every: int = 1,
    skip_final: Optional[int] = None,
) -> Tensor:
    sens, slens = getattr(batch, sen_column)
    batch_size = batch.batch_size

    # Use PackedSequence if sen lens are not all equal
    if len(set(slens.tolist())) > 1:
        packed_sens = pack_padded_sequence(sens, lengths=slens, batch_first=True)

        hidden = model.init_hidden(batch_size)
        final_hidden = torch.zeros(
            (batch_size, model.output_size), dtype=config.DTYPE
        ).to(sens.device)
        n = 0
        if skip_final is not None and skip_final > 0:
            skip_final = -skip_final
        for i, j in enumerate(packed_sens.batch_sizes[:skip_final]):
            w = packed_sens[0][n : n + j]
            for name, v in hidden.items():
                # AWD-LSTM has an extra hidden state dimension
                if hidden[name].size(0) == 1:
                    hidden[name] = v[:, :j]
                else:
                    hidden[name] = v[:j]
            if hasattr(model, "use_char_embs") and model.use_char_embs:
                w = [sen[j] for sen in all_sens]
            with torch.no_grad():
                _, hidden = model(w, hidden, compute_out=False)[:2]
            for k in range(int(j)):
                final_hidden[k] = hidden[model.top_layer, "hx"].squeeze()[k]
            n += j.item()
    else:
        hidden = model.init_hidden(batch_size // skip_every)
        for j in range(slens[0].item() - (skip_final or 0)):
            if hasattr(model, "use_char_embs") and model.use_char_embs:
                w = [sen[j] for sen in all_sens]
            else:
                w = sens[::skip_every, j]
            with torch.no_grad():
                _, hidden = model(w, hidden, compute_out=False)[:2]

        final_hidden = model.final_hidden(hidden)

    return final_hidden


def create_unk_sen_mask(vocab: Vocab, sens: List[List[str]]) -> Tensor:
    """
    Creates a tensor mask for sentences that contain at least one
    word that is not part of the model vocabulary.
    """
    mask = torch.zeros(len(sens), dtype=torch.uint8)
    for i, sen in enumerate(sens):
        for w in sen:
            if w not in vocab.stoi:
                mask[i] = True
                warnings.warn(f"'{w}' is not part of model vocab!")

    return mask
