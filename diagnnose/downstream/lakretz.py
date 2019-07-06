import os
from typing import Any, Dict, List, Optional

from diagnnose.corpora.create_iterator import create_iterator
from diagnnose.corpora.import_corpus import import_corpus
from diagnnose.models.language_model import LanguageModel

task_descriptions: Dict[str, Any] = {
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
    corpus_path: str,
    vocab_path: str,
    lakretz_tasks: Optional[List[str]] = None,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    """ Performs the downstream tasks described in Lakretz et al. (2019)

    Arxiv link: https://arxiv.org/pdf/1903.07435.pdf
    Repo: https://github.com/FAIRNS/Number_and_syntax_units_in_LSTM_LMs

    Parameters
    ----------
    model : LanguageModel
        Language model for which the accuracy is calculated.
    corpus_path : str
        Path to directory containing the the datasets that can be found
        in the github repo.
    vocab_path : str
        Path to vocabulary file of the Language Model.
    lakretz_tasks : List[str], optional
        The downstream tasks that will be tested. If not provided this
        will default to the full set of conditions.
    device : str, optional
        Torch device name on which model will be run. Defaults to cpu.

    Returns
    -------
    accs_dict : Dict[str, Dict[str, float]]
        Dictionary mapping a downstream task to a task condition to the
        model accuracy.
    """
    if lakretz_tasks is None:
        lakretz_tasks = list(task_descriptions.keys())

    accs_dict: Dict[str, Dict[str, float]] = {}

    for task in lakretz_tasks:
        assert (
            task in task_descriptions
        ), f"Provided condition {task} is not recognised!"

        print(task)
        condition_specs = task_descriptions[task]

        corpus = import_corpus(
            os.path.join(corpus_path, f"{task}.txt"),
            corpus_header=["sen", "type", "correct", "idx"],
            vocab_path=vocab_path,
        )

        iterator = create_iterator(corpus, batch_size=2, device=device)

        accs = [0.0] * condition_specs["classes"]

        for i, batch in enumerate(iterator):
            sen = batch.sen[0][0][:-1]

            hidden = None
            for w in sen:
                _, hidden = model(w.view(1), hidden, compute_out=False)

            w1 = batch.sen[0][0][-1]
            w2 = batch.sen[0][1][-1]
            classes = [[w1, w2]]

            probs = hidden[model.num_layers - 1]["hx"][0] @ model.decoder_w[classes].t()
            probs += model.decoder_b[classes]

            if probs[0] > probs[1]:
                accs[i // condition_specs["items_per_class"]] += (
                    1 / condition_specs["items_per_class"]
                )
            if i % condition_specs["items_per_class"] == 0 and i > 0:
                prev_condition = (i // condition_specs["items_per_class"]) - 1

                print(
                    f"{condition_specs['conditions'][prev_condition]}: {accs[prev_condition]:.3f}"
                )

        accs_dict[task] = dict(zip(condition_specs["conditions"], accs))

    return accs_dict
