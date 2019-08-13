import os
from typing import Any, Dict, List, Optional

import torch
from torchtext.data import BucketIterator

from diagnnose.activations.activation_reader import ActivationReader
from diagnnose.corpus.create_iterator import create_iterator
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.models.lm import LanguageModel
from diagnnose.typedefs.corpus import Corpus

marvin_descriptions: Dict[str, Any] = {
    "npi_no_the_ever": {"classes": 1, "items_per_class": 400},
    "npi_no_some_ever": {"classes": 1, "items_per_class": 400},
}


def marvin_init(
    vocab_path: str,
    path: str,
    task_activations: Optional[Dict[str, str]] = None,
    tasks: Optional[List[str]] = None,
    device: str = "cpu",
) -> Dict[str, Dict[str, Any]]:
    """ Performs the initialization for the tasks of
    Marvin & Linzen (2018)

    Arxiv link: https://arxiv.org/pdf/1808.09031.pdf
    Repo: https://github.com/BeckyMarvin/LM_syneval

    Parameters
    ----------
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

    Returns
    -------
    init_dict : Dict[str, Dict[str, Any]]
        Dictionary containing the initial task setup.
    """

    task_activations = task_activations or {}

    if tasks is None:
        tasks = list(marvin_descriptions.keys())

    accs_dict: Dict[str, float] = {}
    activation_readers: Dict[str, Optional[ActivationReader]] = {}
    corpora: Dict[str, Corpus] = {}
    iterators: Dict[str, BucketIterator] = {}

    for task in tasks:
        assert task in marvin_descriptions, f"Provided task {task} is not recognised!"

        activation_dir = task_activations.get(task, None)
        activation_readers[task] = (
            ActivationReader(activation_dir) if activation_dir is not None else None
        )

        task_specs = marvin_descriptions[task]
        items_per_class = task_specs["items_per_class"]

        corpora[task] = import_corpus(
            os.path.join(path, f"{task}.txt"),
            vocab_path=vocab_path,
        )

        iterators[task] = create_iterator(
            corpora[task], batch_size=(items_per_class * 2), device=device
        )

        accs_dict[task] = 0.0

    return {
        "accs_dict": accs_dict,
        "activation_readers": activation_readers,
        "corpora": corpora,
        "iterators": iterators,
    }


def marvin_downstream(
    init_dict: Dict[str, Dict[str, Any]], model: LanguageModel
) -> Dict[str, float]:
    """ Performs the downstream tasks of Marvin & Linzen (2018)

    Arxiv link: https://arxiv.org/pdf/1808.09031.pdf
    Repo: https://github.com/BeckyMarvin/LM_syneval

    Parameters
    ----------
    init_dict : Dict[str, Dict[str, Any]]
        Dictionary created using `marvin_init` containing the initial
        task setup.
    model : LanguageModel
        Language model for which the accuracy is calculated.

    Returns
    -------
    accs_dict : Dict[str, Dict[str, float]]
        Dictionary mapping a downstream task to a task condition to the
        model accuracy.
    """
    for task in init_dict["corpora"].keys():
        activation_reader = init_dict["activation_readers"][task]
        corpus = init_dict["corpora"][task]
        iterator = init_dict["iterators"][task]

        sen = next(iter(iterator)).sen[0]
        batch_size = sen.size(0)

        if activation_reader is None:
            hidden = model.init_hidden(batch_size)
            for j in range(sen.size(1)):
                if model.use_char_embs:
                    tokens = [corpus.examples[i].sen[j] for i in range(batch_size)]
                else:
                    tokens = sen[:, j]
                with torch.no_grad():
                    _, hidden = model(tokens, hidden, compute_out=False)

            final_hidden = model.final_hidden(hidden)
        else:
            activation_index = (
                slice(None, None, None),
                {"a_name": (model.top_layer, "hx")},
            )
            final_hidden = torch.stack(
                [t[-1] for t in activation_reader[activation_index]], 0
            )

        classes = [[corpus.vocab.stoi["ever"]]]

        probs = final_hidden @ model.decoder_w[classes].squeeze()
        probs += model.decoder_b[classes]

        acc = torch.mean((probs[::2] > probs[1::2]).to(torch.float)).item()
        init_dict["accs_dict"][task] = acc

        print(f"{task}: {acc:.3f}")

    return init_dict["accs_dict"]
