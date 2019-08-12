import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from diagnnose.activations.activation_reader import ActivationReader
from diagnnose.corpus.create_iterator import create_iterator
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.models.lm import LanguageModel

lakretz_descriptions: Dict[str, Any] = {
    "adv": {"classes": 2, "items_per_class": 900, "conditions": ["S", "P"]},
    "adv_adv": {"classes": 2, "items_per_class": 900, "conditions": ["S", "P"]},
    "adv_conjunction": {"classes": 2, "items_per_class": 600, "conditions": ["S", "P"]},
    "namepp": {"classes": 2, "items_per_class": 900, "conditions": ["S", "P"]},
    "nounpp": {
        "classes": 4,
        "items_per_class": 600,
        "conditions": ["SS", "SP", "PS", "PP"],
    },
    "nounpp_adv": {
        "classes": 4,
        "items_per_class": 600,
        "conditions": ["SS", "SP", "PS", "PP"],
    },
    "simple": {"classes": 2, "items_per_class": 300, "conditions": ["S", "P"]},
}


def lakretz_downstream(
    model: LanguageModel,
    vocab_path: str,
    path: str,
    task_activations: Optional[Dict[str, str]] = None,
    tasks: Optional[List[str]] = None,
    device: str = "cpu",
    print_results: bool = True,
    ignore_unk: bool = True,
) -> Dict[str, Dict[str, float]]:
    """ Performs the downstream tasks described in Lakretz et al. (2019)

    Arxiv link: https://arxiv.org/pdf/1903.07435.pdf
    Repo: https://github.com/FAIRNS/Number_and_syntax_units_in_LSTM_LMs

    Parameters
    ----------
    model : LanguageModel
        Language model for which the accuracy is calculated.
    vocab_path : str
        Path to vocabulary file of the Language Model.
    path : str
        Path to directory containing the datasets that can be found
        in the github repo.
    task_activations : str, optional
        Dictionary mapping task names to directories to which the
        Lakretz task embeddings have been extracted. If a task is not
        provided the activations will be created during the task.
    tasks : List[str], optional
        The downstream tasks that will be tested. If not provided this
        will default to the full set of conditions.
    device : str, optional
        Torch device name on which model will be run. Defaults to cpu.
    print_results : bool, optional
        Toggle on to print task results directly. Defaults to True.
    ignore_unk : bool, optional
        Ignore cases for which at least one of the cases of the verb
        is not part of the model vocabulary. Defaults to True.

    Returns
    -------
    accs_dict : Dict[str, Dict[str, float]]
        Dictionary mapping a downstream task to a task condition to the
        model accuracy.
    """
    task_activations = task_activations or {}

    if tasks is None:
        tasks = list(lakretz_descriptions.keys())

    accs_dict: Dict[str, Dict[str, float]] = {}

    for task in tasks:
        assert task in lakretz_descriptions, f"Provided task {task} is not recognised!"

        activation_dir = task_activations.get(task, None)
        activation_reader = (
            ActivationReader(activation_dir) if activation_dir is not None else None
        )

        if print_results:
            print(f"\n{task}")
        condition_specs = lakretz_descriptions[task]
        items_per_class = condition_specs["items_per_class"]

        corpus = import_corpus(
            os.path.join(path, f"{task}.txt"),
            corpus_header=["sen", "type", "correct", "idx"],
            vocab_path=vocab_path,
        )

        sen_len = len(corpus.examples[0].sen)
        iterator = create_iterator(
            corpus, batch_size=(items_per_class * 2), device=device
        )

        skipped = 0
        accs = np.zeros(condition_specs["classes"])

        for i, batch in enumerate(iterator):
            if activation_dir is None:
                hidden = model.init_hidden(items_per_class)
                for j in range(sen_len - 1):
                    if model.use_char_embs:
                        w = [
                            corpus.examples[k].sen[j]
                            for k in range(
                                i * items_per_class * 2,
                                (i + 1) * items_per_class * 2,
                                2,
                            )
                        ]
                    else:
                        w = batch.sen[0][::2, j]
                    with torch.no_grad():
                        _, hidden = model(w, hidden, compute_out=False)

                final_hidden = model.final_hidden(hidden)
            else:
                sen_slice = slice(
                    i * items_per_class * 2, (i + 1) * items_per_class * 2, 2
                )
                final_hidden = torch.stack(
                    activation_reader[sen_slice, {"a_name": (model.top_layer, "hx")}],
                    dim=0,
                )[:, -2, :]

            w1: Tensor = batch.sen[0][::2, -1]
            w2: Tensor = batch.sen[0][1::2, -1]

            if ignore_unk:
                unk_idx = corpus.vocab.stoi.unk_idx
                for widx, mask_vals in enumerate(zip((w1 == unk_idx), (w2 == unk_idx))):
                    if mask_vals[0] == 1:
                        token = corpus.examples[widx * 2].sen[-1]
                        warnings.warn(f"'{token}' is not part of model vocab!")
                    if mask_vals[1] == 1:
                        token = corpus.examples[widx * 2 + 1].sen[-1]
                        warnings.warn(f"'{token}' is not part of model vocab!")
                mask = (w1 == unk_idx) | (w2 == unk_idx)
                skipped = int(torch.sum(mask))
                w1 = w1[~mask]
                w2 = w2[~mask]
                final_hidden = final_hidden[~mask]

            classes = torch.stack((w1, w2), dim=1)

            probs = torch.bmm(model.decoder_w[classes], final_hidden.unsqueeze(2))
            probs = probs[:, :, 0]
            probs += model.decoder_b[classes]

            accs[i] += torch.mean((probs[:, 0] >= probs[:, 1]).to(torch.float))

            if print_results:
                print(f"{condition_specs['conditions'][i]}:\t{accs[i]:.3f}")
                if skipped > 0:
                    print(f"{skipped:.0f}/{items_per_class} items were skipped.\n")

        accs_dict[task] = dict(zip(condition_specs["conditions"], accs))

    return accs_dict
