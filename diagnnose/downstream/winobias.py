import os
import shutil
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from diagnnose.corpus.create_iterator import create_iterator
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.decompositions.factory import DecomposerFactory
from diagnnose.models.lm import LanguageModel
from diagnnose.typedefs.corpus import Corpus

from .misc import calc_final_hidden, create_unk_sen_mask

TMP_DIR = "winobias_activations"

winobias_descriptions: Dict[str, Any] = {
    "unamb": {
        "mm": {"path": "unamb_MM.txt"},
        "mf": {"path": "unamb_MF.txt"},
        "fm": {"path": "unamb_FM.txt"},
        "ff": {"path": "unamb_FF.txt"},
    },
    "stereo": {
        "mm": {"path": "stereo_MM.txt"},
        "mf": {"path": "stereo_MF.txt"},
        "fm": {"path": "stereo_FM.txt"},
        "ff": {"path": "stereo_FF.txt"},
    },
}


def winobias_init(
    vocab_path: str,
    path: str,
    task_activations: Optional[str] = None,
    tasks: Optional[List[str]] = None,
    device: str = "cpu",
    save_activations: bool = True,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
    """ Initializes the adapted tasks of Zhao et al. (2018)

    Arxiv link: https://arxiv.org/abs/1804.06876
    Repo: TODO: put adapted files online

    Parameters
    ----------
    vocab_path : str
        Path to vocabulary file of the Language Model.
    path : str
        Path to directory containing the datasets that can be found
        in the github repo.
    task_activations : str, optional
        Path to folder containing the extracted activations. If not
        provided new activations will be extracted.
    tasks : List[str], optional
        The downstream tasks that will be tested. If not provided this
        will default to the full set of conditions.re
    device : str, optional
        Torch device name on which model will be run. Defaults to cpu.
    save_activations : bool, optional
        Toggle to save the extracted activations, otherwise delete them.
        Defaults to True.

    Returns
    -------
    init_dict : Dict[str, Dict[str, Any]]
        Dictionary containing the initial task setup, mapping each task
        to to required fields.
    """

    if tasks is None:
        tasks = list(winobias_descriptions.keys())

    init_dict: Dict[str, Dict[str, Any]] = {}

    for task in tasks:
        init_dict[task] = {}
        for condition in winobias_descriptions[task]:
            assert (
                task in winobias_descriptions
            ), f"Provided task {task} is not recognised!"

            corpus = import_corpus(
                os.path.join(path, winobias_descriptions[task][condition]["path"]),
                header_from_first_line=True,
                vocab_path=vocab_path,
            )

            iterator = create_iterator(corpus, batch_size=500, device=device, sort=True)

            if task_activations is not None:
                activations_dir = os.path.join(task_activations, task, condition)
            else:
                activations_dir = None

            init_dict[task][condition] = {
                "corpus": corpus,
                "iterator": iterator,
                "activations_dir": activations_dir,
                "save_activations": save_activations,
            }

    return init_dict


def winobias_downstream(
    init_dict: Dict[str, Dict[str, Any]],
    model: LanguageModel,
    decompose_config: Optional[Dict[str, Any]] = None,
    ignore_unk: bool = False,
    decomp_obj_idx: bool = False,
    add_dec_bias: bool = True,
) -> Dict[str, Dict[str, float]]:
    """ Performs the adapted tasks of Zhao et al. (2018)

    Arxiv link: https://arxiv.org/abs/1804.06876
    Repo: TODO: put adapted files online

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
    decomp_obj_idx : bool
        Toggle to decompose wrt the object positions. Defaults to False.
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
        task: {condition: 0.0 for condition in winobias_descriptions[task].keys()}
        for task in init_dict.keys()
    }
    for task, init_task in init_dict.items():
        print()
        for condition, init_condition in init_task.items():
            corpus = init_condition["corpus"]
            iterator = init_condition["iterator"]
            activations_dir = init_condition["activations_dir"]
            if decompose_config is not None:
                create_new_activations = False
                if activations_dir is None:
                    activations_dir = os.path.join(TMP_DIR, task, condition)
                    create_new_activations = True
                factory = DecomposerFactory(
                    model,
                    activations_dir,
                    create_new_activations=create_new_activations,
                    corpus=corpus,
                    sen_ids=slice(0, 500),
                )
            else:
                factory = None

            skipped = 0

            for cidx, batch in enumerate(iterator):
                batch_size = batch.batch_size
                all_sens = [ex.sen for ex in corpus.examples]
                if factory is None:
                    final_hidden = calc_final_hidden(model, batch, all_sens)
                else:
                    lens = torch.tensor([len(ex.sen) for ex in corpus]).to(torch.long)
                    decomposer = factory.create(slice(0, batch_size))
                    if decomp_obj_idx:
                        obj_idx_start = torch.tensor(
                            [ex.obj_idx_start - 1 for ex in corpus]
                        )
                        obj_idx_end = torch.tensor([ex.obj_idx + 1 for ex in corpus])
                        decompose_config.update(
                            {"start": obj_idx_start, "stop": obj_idx_end}
                        )
                    final_hidden = decomposer.decompose(**decompose_config)["relevant"][
                        range(500), lens - 1
                    ]

                classes = create_winobias_classes(batch.ref_type, corpus)

                if ignore_unk:
                    mask = create_unk_sen_mask(corpus.vocab, all_sens)
                    skipped = int(torch.sum(mask))
                    classes = classes[~mask]
                    final_hidden = final_hidden[~mask]

                probs = torch.bmm(model.decoder_w[classes], final_hidden.unsqueeze(2))
                probs = probs[:, :, 0]
                if add_dec_bias:
                    probs += model.decoder_b[classes]

                acc = torch.mean((probs[:, 0] >= probs[:, 1]).to(torch.float)).item()
                accuracies[task][condition] = acc

                print(f"{task}.{condition}:\t{acc:.3f}")
                if skipped > 0:
                    print(f"{skipped:.0f}/500 items were skipped.")

            if not init_condition["save_activations"]:
                shutil.rmtree(activations_dir)

    return accuracies


def create_winobias_classes(ref_types: List[str], corpus: Corpus) -> Tensor:
    classes = torch.zeros((len(corpus), 2), dtype=torch.long)

    for i, ref_type in enumerate(ref_types):
        if ref_type == "subj":
            classes[i] = torch.tensor(
                [corpus.vocab.stoi["he"], corpus.vocab.stoi["she"]]
            )
        else:
            classes[i] = torch.tensor(
                [corpus.vocab.stoi["his"], corpus.vocab.stoi["her"]]
            )

    return classes
