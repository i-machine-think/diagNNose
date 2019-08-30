import glob
from typing import Any, Dict, List, Optional

import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torchtext.data import BucketIterator, Dataset, Example, Field, RawField

from diagnnose.corpus.create_iterator import create_iterator
from diagnnose.models.lm import LanguageModel
from diagnnose.typedefs.activations import DTYPE
from diagnnose.typedefs.corpus import Corpus
from diagnnose.utils.pickle import load_pickle
from diagnnose.vocab import attach_vocab


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
    all_paths = glob.glob(f"{path}/*.pickle")
    all_tasks = [path.split("/")[-1].split(".")[0] for path in all_paths]
    task2path = dict(zip(all_tasks, all_paths))

    tasks = tasks or all_tasks

    init_dict: Dict[str, Dict[str, Any]] = {}

    for task in tasks:
        if "npi" in task:
            continue
        corpus_dict = load_pickle(task2path[task])

        accs_dict: Dict[str, float] = {}
        corpora: Dict[str, Corpus] = {}
        iterators: Dict[str, BucketIterator] = {}
        fields = [
            ("sen", Field(batch_first=True, include_lengths=True)),
            ("postfix", RawField()),
            ("idx", RawField()),
        ]
        fields[1][1].is_target = False
        fields[2][1].is_target = False

        for condition, sens in corpus_dict.items():
            examples = []
            if len(sens[0]) > 2:
                print(sens[0])
            for idx, (s1, s2) in enumerate(sens):
                s1, s2 = s1.split(), s2.split()

                verb_index = -1
                for _ in range(len(s1)):
                    if s1[:verb_index] == s2[:verb_index]:
                        break
                    verb_index -= 1
                assert -verb_index < len(s1)

                # None slice selects full sentence, otherwise select sentence till verb
                subsen = s1[: (verb_index + 1 or None)]
                ex = Example.fromlist(
                    [
                        subsen + [s2[verb_index]],  # sen + wrong verb
                        s1[len(s1) + verb_index + 1 : len(s1)],  # postfix
                        idx,  # sen idx
                    ],
                    fields,
                )
                examples.append(ex)
            corpus = Dataset(examples, fields)
            attach_vocab(corpus, vocab_path)
            corpora[condition] = corpus
            iterators[condition] = create_iterator(
                corpus, batch_size=len(sens), device=device, sort=True
            )
            accs_dict[condition] = 0.0

        init_dict[task] = {
            "accs_dict": accs_dict,
            # "activation_readers": activation_readers,
            "corpora": corpora,
            "iterators": iterators,
        }

    return init_dict


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
    for task, init_task in init_dict.items():
        for condition in init_task["corpora"].keys():
            corpus = init_task["corpora"][condition]
            iterator = init_task["iterators"][condition]

            batch = next(iter(iterator))
            sens, slens = batch.sen
            batch_size = batch.batch_size
            packed_sens = pack_padded_sequence(sens, lengths=slens, batch_first=True)

            hidden = model.init_hidden(batch_size)
            final_hidden = torch.zeros((batch_size, model.output_size), dtype=DTYPE).to(
                sens.device
            )
            n = 0
            for i, j in enumerate(packed_sens.batch_sizes[:-2]):
                w = packed_sens[0][n : n + j]
                for name, v in hidden.items():
                    hidden[name] = v[:j]
                if model.use_char_embs:
                    w = [corpus.examples[batch.idx[k]].sen[i] for k in range(j)]
                _, hidden = model(w, hidden, compute_out=False)
                for k in range(int(j)):
                    final_hidden[k] = hidden[model.top_layer, "hx"][k]
                n += j

            classes = torch.stack(
                [sens[i, slen - 2 : slen] for i, slen in enumerate(slens)]
            )

            probs = torch.bmm(model.decoder_w[classes], final_hidden.unsqueeze(2))
            probs = probs[:, :, 0]
            probs += model.decoder_b[classes]

            acc = torch.mean((probs[:, 0] >= probs[:, 1]).to(torch.float)).item()

            init_task["accs_dict"][condition] = (acc, batch_size)

        task_size = sum(v[1] for v in init_task["accs_dict"].values())
        mean_acc = sum(v[0] * v[1] for v in init_task["accs_dict"].values()) / task_size
        init_task["accs_dict"]["mean_acc"] = mean_acc
        print(f"{task}: {mean_acc:.3f}")

    return {task: init_dict[task]["accs_dict"] for task in init_dict}
