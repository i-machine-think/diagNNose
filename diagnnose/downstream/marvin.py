import os
from typing import Any, Dict, List, Optional

import torch

from diagnnose.corpus.create_iterator import create_iterator
from diagnnose.corpus.import_corpus import import_corpus
from diagnnose.typedefs.lm import LanguageModel

marvin_descriptions: Dict[str, Any] = {
    "npi_no_the_ever": {"classes": 1},
    "npi_no_some_ever": {"classes": 1},
}


def marvin_downstream(
    model: LanguageModel,
    vocab_path: str,
    marvin_path: str,
    marvin_tasks: Optional[List[str]] = None,
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
    marvin_path : str
        Path to directory containing the Marvin datasets that can be
        found in the github repo.
    marvin_tasks : List[str], optional
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
    if marvin_tasks is None:
        marvin_tasks = list(marvin_descriptions.keys())

    accs_dict: Dict[str, float] = {}

    for task in marvin_descriptions:
        assert task in marvin_tasks, f"Provided task {task} is not recognised!"

        corpus = import_corpus(
            os.path.join(marvin_path, f"{task}.txt"), vocab_path=vocab_path
        )

        iterator = create_iterator(corpus, batch_size=2, device=device)

        acc = 0.0

        for batch in iterator:
            sen = batch.sen[0]

            hidden = model.init_hidden(2)
            for i in range(sen.size(1)):
                with torch.no_grad():
                    _, hidden = model(sen[:, i], hidden, compute_out=False)

            classes = [[corpus.vocab.stoi["ever"]]]

            final_hidden = model.final_hidden(hidden)
            probs = final_hidden @ model.decoder_w[classes].squeeze()
            probs += model.decoder_b[classes]

            if probs[0] > probs[1]:
                acc += 1 / len(iterator)

        accs_dict[task] = acc
        if print_results:
            print(f"{task}: {acc:.3f}")

    return accs_dict
