from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torch.nn.functional import log_softmax
from torchtext.data import Example

from diagnnose.corpus import Corpus
from diagnnose.corpus.create_iterator import create_iterator
from diagnnose.typedefs.models import LanguageModel

from ..misc import calc_final_hidden, create_unk_sen_mask
from .preproc import ENVS, create_downstream_corpus, preproc_warstadt


def warstadt_init(
    vocab_path: str,
    path: str,
    subtasks: Optional[List[str]] = None,
    device: str = "cpu",
    use_full_model_probs: bool = True,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """ Initializes the tasks described in Warstadt et al. (2019)

    Paper: https://arxiv.org/pdf/1901.03438.pdf
    Data: https://alexwarstadt.files.wordpress.com/2019/08/npi_lincensing_data.zip

    Parameters
    ----------
    vocab_path : str
        Path to vocabulary file of the Language Model.
    path : str
        Path to the original corpus.
    subtasks : List[str], optional
        The licensing environments that will be tested. If not provided
        this will default to the full set of environments.
    device : str, optional
        Torch device name on which model will be run. Defaults to cpu.
    use_full_model_probs : bool, optional
        Toggle to calculate the full model probs for the NPI sentences.
        If set to False only the NPI logits will be compared, instead
        of their Softmax probabilities. Defaults to True.

    Returns
    -------
    init_dict : Dict[str, Dict[str, Any]]
        Dictionary containing the initial env setup, mapping each env
        to to required fields.
    """
    if subtasks is None:
        subtasks = ENVS

    init_dict: Dict[str, Dict[str, Any]] = {}

    orig_corpus = preproc_warstadt(path)[0]

    for env in subtasks:
        assert env in ENVS, f"Provided env {env} is not recognised!"

        raw_corpus = create_downstream_corpus(orig_corpus, envs=[env])

        header = raw_corpus[0].split("\t")
        tokenize_columns = ["sen", "counter_sen"]
        fields = Corpus.create_fields(header, tokenize_columns=tokenize_columns)
        examples = [
            Example.fromlist(line.split("\t"), fields.items()) for line in raw_corpus
        ]
        corpus = Corpus(
            examples, fields, vocab_path=vocab_path, tokenize_columns=tokenize_columns
        )

        batch_size = 30 if use_full_model_probs else len(corpus)

        iterator = create_iterator(
            corpus, batch_size=batch_size, device=device, sort=True
        )

        init_dict[env] = {"corpus": corpus, "iterator": iterator}

    return init_dict


def warstadt_downstream(
    init_dict: Dict[str, Dict[str, Any]],
    model: LanguageModel,
    ignore_unk: bool = True,
    use_full_model_probs: bool = False,
    **kwargs: Any,
) -> Dict[str, Dict[str, float]]:
    """ Performs the downstream tasks described in Warstadt et al. (2019)

    Paper: https://arxiv.org/pdf/1901.03438.pdf
    Data: https://alexwarstadt.files.wordpress.com/2019/08/npi_lincensing_data.zip

    Parameters
    ----------
    init_dict : Dict[str, Dict[str, Any]]
        Dictionary created using `warstadt_init` containing the initial
        task setup.
    model : LanguageModel
        Language model for which the accuracy is calculated.
    ignore_unk : bool, optional
        Ignore cases for which at least one of the cases of the verb
        is not part of the model vocabulary. Defaults to False.
    use_full_model_probs : bool, optional
        Toggle to calculate the full model probs for the NPI sentences.
        If set to False only the NPI logits will be compared, instead
        of their Softmax probabilities. Defaults to True.

    Returns
    -------
    accs_dict : Dict[str, float]
        Dictionary mapping a licensing env to the model accuracy.
    """
    accuracies = {env: 0.0 for env in init_dict.keys()}
    for env, init_env in init_dict.items():
        print(f"\n{env}")
        corpus = init_env["corpus"]
        iterator = init_env["iterator"]

        skipped = 0

        for batch in iterator:
            all_sens = [ex.sen for ex in corpus.examples]
            final_hidden = calc_final_hidden(model, batch, all_sens)
            wfinal_hidden = calc_final_hidden(
                model, batch, all_sens, sort_sens=True, sen_column="counter_sen"
            )

            npi_ids = torch.tensor([corpus.vocab.stoi[npi] for npi in batch.npi])

            if use_full_model_probs:
                classes = torch.tensor(list(corpus.vocab.stoi.values()))
            else:
                classes = npi_ids.unsqueeze(1)

            if ignore_unk:
                # We base our mask on the correct sentences and apply that to both cases
                mask = create_unk_sen_mask(corpus.vocab, all_sens)
                skipped += int(torch.sum(mask))
                classes = classes[~mask]
                final_hidden = final_hidden[~mask]
                wfinal_hidden = wfinal_hidden[~mask]

            acc = calc_acc(
                model,
                classes,
                final_hidden,
                wfinal_hidden,
                npi_ids,
                use_full_model_probs,
                batch.batch_size,
            )
            accuracies[env] += acc * batch.batch_size

        accuracies[env] /= len(corpus)
        print(f"{env}:\t{accuracies[env]:.3f}")
        if skipped > 0:
            print(f"{skipped:.0f}/{len(corpus)} items were skipped.\n")

    return accuracies


def calc_acc(
    model: LanguageModel,
    classes: Tensor,
    final_hidden: Tensor,
    wfinal_hidden: Tensor,
    npi_ids: Tensor,
    use_full_model_probs: bool,
    batch_size: int,
) -> float:
    decoder_w = model.decoder_w[classes]
    decoder_b = model.decoder_b[classes].squeeze()

    if use_full_model_probs:
        if batch_size == 1:
            final_hidden = final_hidden.unsqueeze(0)
            wfinal_hidden = wfinal_hidden.unsqueeze(0)

        # Calculate model logits
        logits = final_hidden @ decoder_w.t() + decoder_b
        wlogits = wfinal_hidden @ decoder_w.t() + decoder_b

        # Retrieve SoftMax probabilities
        probs = log_softmax(logits, dim=1)
        wprobs = log_softmax(wlogits, dim=1)

        # Select sentence-specific NPI probabilities
        probs = probs[range(batch_size), npi_ids]
        wprobs = wprobs[range(batch_size), npi_ids]
    else:
        probs = torch.bmm(decoder_w, final_hidden.unsqueeze(2)).squeeze() + decoder_b
        wprobs = torch.bmm(decoder_w, wfinal_hidden.unsqueeze(2)).squeeze() + decoder_b

    acc = torch.mean((probs >= wprobs).to(torch.float)).item()

    return acc
