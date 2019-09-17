import os
import shutil
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from diagnnose.activations.activation_reader import ActivationReader
from diagnnose.corpus.create_iterator import create_iterator
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.decompositions.factory import DecomposerFactory
from diagnnose.models.lm import LanguageModel

from .misc import calc_final_hidden, create_unk_sen_mask

TMP_DIR = "lakretz_activations"

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
            header=["sen", "type", "correct", "idx"],
            vocab_path=vocab_path,
        )

        iterator = create_iterator(
            corpus, batch_size=(items_per_class * 2), device=device
        )

        init_dict[task] = {
            "activation_reader": activation_reader,
            "corpus": corpus,
            "iterator": iterator,
        }

    return init_dict


def lakretz_downstream(
    init_dict: Dict[str, Dict[str, Any]],
    model: LanguageModel,
    decompose_config: Optional[Dict[str, Any]] = None,
    ignore_unk: bool = True,
    add_dec_bias: bool = True,
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
    decompose_config : Dict[str, Any], optional
        Config dict containing setup for contextual decompositions.
        If provided the decomposed predictions will be used for the
        task. Note that `task_activations` should be passed as well, in
        the `downstream.config.lakretz` object that is passed to the
        downstream suite.
    ignore_unk : bool, optional
        Ignore cases for which at least one of the cases of the verb
        is not part of the model vocabulary. Defaults to True.
    add_dec_bias : bool
        Toggle to add the decoder bias to the score that is compared.
        Defaults to True.

    Returns
    -------
    accs_dict : Dict[str, Dict[str, float]]
        Dictionary mapping a downstream task to a task condition to the
        model accuracy.
    """
    accuracies = {
        task: {condition: 0.0 for condition in lakretz_descriptions[task]["conditions"]}
        for task in init_dict.keys()
    }
    for task, init_task in init_dict.items():
        print(f"\n{task}")
        corpus = init_task["corpus"]
        iterator = init_task["iterator"]
        activation_reader = init_task["activation_reader"]
        activations_dir = os.path.join(TMP_DIR, task)
        if decompose_config is not None:
            if activation_reader is not None:
                activations_dir = activation_reader.activations_dir
            factory = DecomposerFactory(
                model,
                activations_dir,
                create_new_activations=(activation_reader is None),
                corpus=corpus,
            )
        else:
            factory = None

        skipped = 0
        task_specs = lakretz_descriptions[task]
        items_per_class = task_specs["items_per_class"]

        for cidx, batch in enumerate(iterator):
            all_sens = [
                corpus.examples[k].sen
                for k in range(
                    cidx * items_per_class * 2, (cidx + 1) * items_per_class * 2
                )
            ]
            if activation_reader is None and factory is None:
                final_hidden = calc_final_hidden(
                    model, batch, all_sens[::2], skip_every=2, skip_final=1
                )
            else:
                sen_slice = slice(
                    cidx * items_per_class * 2, (cidx + 1) * items_per_class * 2, 2
                )
                if factory is not None:
                    decomposer = factory.create(sen_slice, slice(0, -1))
                    final_hidden = decomposer.decompose(**decompose_config)["relevant"][
                        :, -1
                    ]
                else:
                    final_hidden = torch.stack(
                        activation_reader[
                            sen_slice, {"a_name": (model.top_layer, "hx")}
                        ],
                        dim=0,
                    )[:, -2, :]

            w1: Tensor = batch.sen[0][::2, -1]
            w2: Tensor = batch.sen[0][1::2, -1]
            classes = torch.stack((w1, w2), dim=1)

            if ignore_unk:
                mask = create_unk_sen_mask(corpus.vocab, all_sens)
                mask = mask[::2] | mask[1::2]
                skipped = int(torch.sum(mask))
                classes = classes[~mask]
                final_hidden = final_hidden[~mask]

            probs = torch.bmm(model.decoder_w[classes], final_hidden.unsqueeze(2))
            probs = probs[:, :, 0]
            if add_dec_bias:
                probs += model.decoder_b[classes]

            acc = torch.mean((probs[:, 0] >= probs[:, 1]).to(torch.float)).item()
            accuracies[task][task_specs["conditions"][cidx]] = acc

            print(f"{task_specs['conditions'][cidx]}:\t{acc:.3f}")
            if skipped > 0:
                print(f"{skipped:.0f}/{items_per_class} items were skipped.\n")

        if decompose_config is not None and activation_reader is None:
            shutil.rmtree(activations_dir)

    return accuracies
