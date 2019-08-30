import os
import warnings
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor
from torchtext.data import Batch

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


def lakretz_init(
    vocab_path: str,
    path: str,
    task_activations: Optional[Dict[str, str]] = None,
    tasks: Optional[List[str]] = None,
    device: str = "cpu",
) -> Dict[str, Dict[str, Any]]:
    """ Initializes the tasks described in Lakretz et al. (2019)

    Arxiv link: https://arxiv.org/pdf/1903.07435.pdf
    Repo: https://github.com/FAIRNS/Number_and_syntax_units_in_LSTM_LMs

    Parameters
    ----------
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

    Returns
    -------
    init_dict : Dict[str, Dict[str, Any]]
        Dictionary containing the initial task setup, mapping each task
        to to required fields.
    """

    task_activations = task_activations or {}

    if tasks is None:
        tasks = list(lakretz_descriptions.keys())

    init_dict: Dict[str, Dict[str, Any]] = {}

    for task in tasks:
        assert task in lakretz_descriptions, f"Provided task {task} is not recognised!"

        activation_dir = task_activations.get(task, None)
        activation_reader = (
            ActivationReader(activation_dir) if activation_dir is not None else None
        )

        task_specs = lakretz_descriptions[task]
        items_per_class = task_specs["items_per_class"]

        corpus = import_corpus(
            os.path.join(path, f"{task}.txt"),
            corpus_header=["sen", "type", "correct", "idx"],
            vocab_path=vocab_path,
        )

        iterator = create_iterator(
            corpus, batch_size=(items_per_class * 2), device=device
        )

        accs_dict = {condition: 0.0 for condition in task_specs["conditions"]}

        init_dict[task] = {
            "accs_dict": accs_dict,
            "activation_reader": activation_reader,
            "corpus": corpus,
            "iterator": iterator,
        }

    return init_dict


def lakretz_downstream(
    init_dict: Dict[str, Dict[str, Any]], model: LanguageModel, ignore_unk: bool = True
) -> Dict[str, Dict[str, float]]:
    """ Performs the downstream tasks described in Lakretz et al. (2019)

    Arxiv link: https://arxiv.org/pdf/1903.07435.pdf
    Repo: https://github.com/FAIRNS/Number_and_syntax_units_in_LSTM_LMs

    Parameters
    ----------
    init_dict : Dict[str, Dict[str, Any]]
        Dictionary created using `lakretz_init` containing the initial
        task setup.
    model : LanguageModel
        Language model for which the accuracy is calculated.
    ignore_unk : bool, optional
        Ignore cases for which at least one of the cases of the verb
        is not part of the model vocabulary. Defaults to True.

    Returns
    -------
    accs_dict : Dict[str, Dict[str, float]]
        Dictionary mapping a downstream task to a task condition to the
        model accuracy.
    """
    for task, init_task in init_dict.items():
        print(f"\n{task}")
        activation_reader = init_task["activation_reader"]
        accuracies = init_task["accs_dict"]
        corpus = init_task["corpus"]
        iterator = init_task["iterator"]

        skipped = 0
        task_specs = lakretz_descriptions[task]
        items_per_class = task_specs["items_per_class"]
        sen_len = len(corpus.examples[0].sen)

        for cidx, batch in enumerate(iterator):
            all_sens = [
                corpus.examples[k].sen
                for k in range(
                    cidx * items_per_class * 2, (cidx + 1) * items_per_class * 2
                )
            ]
            if activation_reader is None:
                final_hidden = calc_final_hidden(model, batch, sen_len, all_sens)
            else:
                sen_slice = slice(
                    cidx * items_per_class * 2, (cidx + 1) * items_per_class * 2, 2
                )
                final_hidden = torch.stack(
                    activation_reader[sen_slice, {"a_name": (model.top_layer, "hx")}],
                    dim=0,
                )[:, -2, :]

            w1: Tensor = batch.sen[0][::2, -1]
            w2: Tensor = batch.sen[0][1::2, -1]

            if ignore_unk:
                mask = torch.zeros(items_per_class, dtype=torch.uint8)
                for i, sen in enumerate(all_sens):
                    for w in sen:
                        if w not in corpus.vocab.stoi:
                            mask[i // 2] = True
                            warnings.warn(f"'{w}' is not part of model vocab!")
                skipped = int(torch.sum(mask))
                w1 = w1[~mask]
                w2 = w2[~mask]
                final_hidden = final_hidden[~mask]

            classes = torch.stack((w1, w2), dim=1)

            probs = torch.bmm(model.decoder_w[classes], final_hidden.unsqueeze(2))
            probs = probs[:, :, 0]
            probs += model.decoder_b[classes]

            acc = torch.mean((probs[:, 0] >= probs[:, 1]).to(torch.float)).item()
            accuracies[task_specs["conditions"][cidx]] = acc

            print(f"{task_specs['conditions'][cidx]}:\t{acc:.3f}")
            if skipped > 0:
                print(f"{skipped:.0f}/{items_per_class} items were skipped.\n")

    return {task: init_dict[task]["accs_dict"] for task in init_dict}


def calc_final_hidden(
    model: LanguageModel, batch: Batch, sen_len: int, all_sens: List[List[str]]
) -> Tensor:
    hidden = model.init_hidden(batch.batch_size // 2)
    for j in range(sen_len - 1):
        if model.use_char_embs:
            w = [sen[j] for sen in all_sens[::2]]
        else:
            w = batch.sen[0][::2, j]
        with torch.no_grad():
            _, hidden = model(w, hidden, compute_out=False)

    final_hidden = model.final_hidden(hidden)

    return final_hidden
