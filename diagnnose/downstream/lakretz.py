import os
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import torch

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
    lakretz_path: str,
    lakretz_activations: Optional[Dict[str, str]] = None,
    lakretz_tasks: Optional[List[str]] = None,
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
    lakretz_path : str
        Path to directory containing the datasets that can be found
        in the github repo.
    lakretz_activations : str, optional
        Dictionary mapping task names to directories to which the
        Lakretz task embeddings have been extracted. If a task is not
        provided the activations will be created during the task.
    lakretz_tasks : List[str], optional
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
    lakretz_activations = lakretz_activations or {}

    if lakretz_tasks is None:
        lakretz_tasks = list(lakretz_descriptions.keys())

    accs_dict: Dict[str, Dict[str, float]] = {}

    for task in lakretz_tasks:
        assert task in lakretz_descriptions, f"Provided task {task} is not recognised!"

        activation_dir = lakretz_activations.get(task, None)
        activation_reader = (
            ActivationReader(activation_dir) if activation_dir is not None else None
        )

        if print_results:
            print(task)
        condition_specs = lakretz_descriptions[task]
        items_per_class = condition_specs["items_per_class"]

        corpus = import_corpus(
            os.path.join(lakretz_path, f"{task}.txt"),
            corpus_header=["sen", "type", "correct", "idx"],
            vocab_path=vocab_path,
        )

        iterator = create_iterator(corpus, batch_size=2, device=device)

        skipped = 0
        accs = np.zeros(condition_specs["classes"])
        tots = np.zeros(condition_specs["classes"])

        for i, batch in enumerate(iterator):
            if print_results and i % items_per_class == 1 and i > 1:
                prev_condition = ((i - 1) // items_per_class) - 1

                prev_acc = accs[prev_condition] / tots[prev_condition]
                skipped = items_per_class - tots[prev_condition]
                print(
                    f"{condition_specs['conditions'][prev_condition]}:\t{prev_acc:.3f}"
                )
                if skipped:
                    print(f"{skipped:.0f}/{items_per_class} items were skipped.")

            if activation_dir is None:
                if model.use_char_embs:
                    sen = corpus.examples[i * 2].sen[:-1]
                else:
                    sen = batch.sen[0][0][:-1]

                hidden = model.init_hidden(1)
                for w in sen:
                    w = [w] if model.use_char_embs else w.view(1)
                    with torch.no_grad():
                        _, hidden = model(w, hidden, compute_out=False)

                final_hidden = model.final_hidden(hidden)
            else:
                assert activation_reader is not None  # mypy being annoying
                final_hidden = activation_reader[
                    i * 2, {"a_name": (model.num_layers - 1, "hx")}
                ][0][-2]

            w1 = batch.sen[0][0][-1]
            w2 = batch.sen[0][1][-1]

            if ignore_unk:
                if w1 == corpus.vocab.stoi.unk_idx:
                    token = corpus.examples[i * 2].sen[-1]
                    warnings.warn(f"'{token}' is not part of model vocab!")
                    continue
                if w2 == corpus.vocab.stoi.unk_idx:
                    token = corpus.examples[i * 2 + 1].sen[-1]
                    warnings.warn(f"'{token}' is not part of model vocab!")
                    continue

            classes = [[w1, w2]]

            probs = model.decoder_w[classes] @ final_hidden
            probs += model.decoder_b[classes]

            if probs[0] >= probs[1]:
                accs[i // items_per_class] += 1
            tots[i // items_per_class] += 1

        if print_results:
            print(f"{condition_specs['conditions'][-1]}:\t{accs[-1]/tots[-1]:.3f}\n")
            if skipped:
                print(f"{skipped:.0f}/{items_per_class} items were skipped.")
        accs_dict[task] = dict(zip(condition_specs["conditions"], accs / tots))

    return accs_dict
