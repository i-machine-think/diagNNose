import os
from typing import Any, Dict, List, Optional

import torch

from diagnnose.activations.activation_reader import ActivationReader
from diagnnose.corpus.create_iterator import create_iterator
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.models.lm import LanguageModel

marvin_descriptions: Dict[str, Any] = {
    "npi_no_the_ever": {"classes": 1},
    "npi_no_some_ever": {"classes": 1},
}


def marvin_downstream(
    model: LanguageModel,
    vocab_path: str,
    path: str,
    task_activations: Optional[Dict[str, str]] = None,
    tasks: Optional[List[str]] = None,
    device: str = "cpu",
    print_results: bool = True,
) -> Dict[str, float]:
    """ Performs the downstream tasks of Marvin & Linzen (2018)

    Arxiv link: https://arxiv.org/pdf/1808.09031.pdf
    Repo: https://github.com/BeckyMarvin/LM_syneval

    Parameters
    ----------
    model : LanguageModel
        Language model for which the accuracy is calculated.
    vocab_path : str
        Path to vocabulary file of the Language Model.
    path : str
        Path to directory containing the Marvin datasets that can be
        found in the github repo.
    task_activations : str, optional
        Dictionary mapping task names to directories to which the
        Marvin task embeddings have been extracted. If a task is not
        provided the activations will be created during the task.
    tasks : List[str], optional
        The downstream tasks that will be tested. If not provided this
        will default to the full set of conditions.
    device : str, optional
        Torch device name on which model will be run. Defaults to cpu.
    print_results : bool, optional
        Toggle on to print task results directly. Defaults to True.

    Returns
    -------
    accs_dict : Dict[str, Dict[str, float]]
        Dictionary mapping a downstream task to a task condition to the
        model accuracy.
    """
    task_activations = task_activations or {}

    if tasks is None:
        tasks = list(marvin_descriptions.keys())

    accs_dict: Dict[str, float] = {}

    for task in tasks:
        assert task in marvin_descriptions, f"Provided task {task} is not recognised!"

        activation_dir = task_activations.get(task, None)
        activation_reader = (
            ActivationReader(activation_dir) if activation_dir is not None else None
        )

        corpus = import_corpus(
            os.path.join(path, f"{task}.txt"), vocab_path=vocab_path
        )

        iterator = create_iterator(corpus, batch_size=2, device=device)

        acc = 0.0

        for i, batch in enumerate(iterator):
            sen = batch.sen[0]

            if activation_dir is None:
                hidden = model.init_hidden(2)
                for j in range(sen.size(1)):
                    if model.use_char_embs:
                        tokens = [
                            corpus.examples[i * 2].sen[j],
                            corpus.examples[i * 2 + 1].sen[j],
                        ]
                    else:
                        tokens = sen[:, j]
                    with torch.no_grad():
                        _, hidden = model(tokens, hidden, compute_out=False)

                final_hidden = model.final_hidden(hidden)
            else:
                assert activation_reader is not None  # mypy being annoying
                layer = model.num_layers - 1
                sen_ids = [i * 2, i * 2 + 1]
                final_hidden = torch.stack(
                    [
                        t[-1]
                        for t in activation_reader[sen_ids, {"a_name": (layer, "hx")}]
                    ],
                    0,
                )

            classes = [[corpus.vocab.stoi["ever"]]]

            probs = final_hidden @ model.decoder_w[classes].squeeze()
            probs += model.decoder_b[classes]

            if probs[0] > probs[1]:
                acc += 1 / len(iterator)

        accs_dict[task] = acc
        if print_results:
            print(f"{task}: {acc:.3f}")

    return accs_dict
