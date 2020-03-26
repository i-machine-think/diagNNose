import glob
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.functional import log_softmax
from torchtext.data import BucketIterator, Dataset, Example, Field, RawField

from diagnnose.corpus.create_iterator import create_iterator
from diagnnose.typedefs.corpus import Corpus
from diagnnose.typedefs.models import LanguageModel
from diagnnose.utils.pickle import load_pickle
from diagnnose.vocab import attach_vocab

from .misc import calc_final_hidden, create_unk_sen_mask


def marvin_init(
    vocab_path: str,
    path: str,
    subtasks: Optional[List[str]] = None,
    device: str = "cpu",
    use_full_model_probs: bool = False,
    **kwargs: Any,
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
    subtasks : List[str], optional
        The downstream tasks that will be tested. If not provided this
        will default to the full set of conditions.
    device : str, optional
        Torch device name on which model will be run. Defaults to cpu.
    use_full_model_probs : bool, optional
        Toggle to calculate the full model probs for the NPI sentences.
        Defaults to False.

    Returns
    -------
    init_dict : Dict[str, Dict[str, Any]]
        Dictionary containing the initial task setup.
    """
    all_paths = glob.glob(f"{path}/*.pickle")
    all_tasks = [path.split("/")[-1].split(".")[0] for path in all_paths]
    task2path = dict(zip(all_tasks, all_paths))

    subtasks = subtasks or all_tasks

    init_dict: Dict[str, Dict[str, Any]] = {}

    for task in subtasks:
        corpus_dict = load_pickle(task2path[task])

        if "npi" in task:
            fields = [
                ("sen", Field(batch_first=True, include_lengths=True)),
                ("wsen", Field(batch_first=True, include_lengths=True)),
                ("postfix", RawField()),
                ("idx", RawField()),
            ]
            fields[2][1].is_target = False
            fields[3][1].is_target = False
        else:
            fields = [
                ("sen", Field(batch_first=True, include_lengths=True)),
                ("postfix", RawField()),
                ("idx", RawField()),
            ]
            fields[1][1].is_target = False
            fields[2][1].is_target = False

        corpora: Dict[str, Corpus] = {}
        iterators: Dict[str, BucketIterator] = {}

        def create_corpus(
            sens_: List[List[str]], condition_: str, batch_size_: int
        ) -> None:
            examples = create_examples(task, sens_, fields)
            corpus = Dataset(examples, fields)
            attach_vocab(corpus, vocab_path)
            if "npi" in task:
                attach_vocab(corpus, vocab_path, sen_column="wsen")
            corpora[condition_] = corpus
            iterators[condition_] = create_iterator(
                corpus, batch_size=batch_size_, device=device, sort=True
            )

        for condition, sens in corpus_dict.items():
            if "npi" in task:
                for i, licensor in enumerate(["no", "few"]):
                    clen = len(sens) // 2
                    sens_subset = sens[i * clen : (i + 1) * clen]
                    batch_size = 20 if use_full_model_probs else len(sens_subset)
                    create_corpus(sens_subset, f"{condition}_{licensor}", batch_size)
            else:
                create_corpus(sens, condition, len(sens))

        init_dict[task] = {"corpora": corpora, "iterators": iterators}

    return init_dict


def create_examples(
    task: str, sens: List[List[str]], fields: List[Tuple[str, Field]]
) -> List[Example]:
    examples = []
    prefixes = set()
    for idx, s in enumerate(sens):
        if "npi" in task:
            s1, s2, _ = s
            s1, s2 = s1.split(), s2.split()
            ever_idx = s1.index("ever")
            prefix = " ".join(s1[:ever_idx])

            # We only need to compute probs for sentences with a unique
            # subsentence prior to the position of the NPI.
            if prefix not in prefixes:
                ex = Example.fromlist(
                    [
                        s1[:ever_idx],  # sen
                        s2[:ever_idx],  # wrong sen
                        s1[ever_idx:],  # postfix
                        len(prefixes),  # sen idx
                    ],
                    fields,
                )
                prefixes.add(prefix)
                examples.append(ex)
        else:
            s1, s2 = s
            s1, s2 = s1.split(), s2.split()

            verb_index = -1
            for _ in range(len(s1)):
                if s1[:verb_index] == s2[:verb_index]:
                    break
                verb_index -= 1
            assert -verb_index < len(s1)

            # None slice selects full sentence (i.e. when verb_index is at eos (-1)),
            # otherwise select sentence till index of the verb
            subsen = s1[: (verb_index + 1 or None)]
            wrong_verb = [s2[verb_index]]
            postfix = s1[len(s1) + verb_index + 1 : len(s1)]
            ex = Example.fromlist([subsen + wrong_verb, postfix, idx], fields)
            examples.append(ex)

    return examples


def marvin_downstream(
    init_dict: Dict[str, Dict[str, Any]],
    model: LanguageModel,
    use_full_model_probs: bool = False,
    ignore_unk: bool = True,
    **kwargs: Any,
) -> Dict[str, Dict[str, Any]]:
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
    use_full_model_probs : bool, optional
        Toggle to calculate the full model probs for the NPI sentences.
        Defaults to False.
    ignore_unk : bool, optional
        Ignore cases for which at least one of the cases of the verb
        is not part of the model vocabulary. Defaults to True.

    Returns
    -------
    accs_dict : Dict[str, Dict[str, float]]
        Dictionary mapping a downstream task to a task condition to the
        model accuracy.
    """
    accuracies: Dict[str, Dict[str, Any]] = {}
    for task, init_task in init_dict.items():
        accuracies[task] = {}
        for condition in init_task["corpora"].keys():
            corpus = init_task["corpora"][condition]
            iterator = init_task["iterators"][condition]
            accs = []
            for batch in iterator:
                batch_size = batch.batch_size

                # Calculate batched hidden states
                if "npi" in task:
                    all_sens = [corpus.examples[idx].sen for idx in batch.idx]

                    final_hidden = calc_final_hidden(model, batch, all_sens)
                    wfinal_hidden = calc_final_hidden(
                        model, batch, all_sens, sen_column="wsen"
                    )
                    if use_full_model_probs:
                        classes = torch.tensor(
                            [list(corpus.vocab.stoi.values())] * batch_size
                        )
                    else:
                        classes = torch.tensor(
                            [corpus.vocab.stoi["ever"]] * batch_size
                        ).unsqueeze(1)
                else:
                    sens, slens = batch.sen
                    all_sens = [corpus.examples[idx].sen for idx in batch.idx]
                    # The final 2 positions of each sentence contain the 2 verb forms
                    final_hidden = calc_final_hidden(
                        model, batch, all_sens, skip_final=2
                    )
                    wfinal_hidden = None

                    classes = torch.stack(
                        [sens[i, slen - 2 : slen] for i, slen in enumerate(slens)]
                    )

                # Mask sentences containing <unk> tokens
                if ignore_unk:
                    mask = create_unk_sen_mask(corpus.vocab, all_sens)
                    skipped = int(torch.sum(mask))
                    if skipped > 0:
                        print(f"{skipped:.0f}/{batch_size} items were skipped.\n")
                    classes = classes[~mask]
                    final_hidden = final_hidden[~mask]
                    if "npi" in task:
                        wfinal_hidden = wfinal_hidden[~mask]

                acc = calc_acc(
                    task,
                    model,
                    corpus,
                    classes,
                    final_hidden,
                    use_full_model_probs,
                    wfinal_hidden,
                )
                accs.append((acc, batch_size))
            accuracies[task][condition] = (
                sum(a[0] * a[1] for a in accs) / len(corpus),
                len(corpus),
            )

        task_size = sum(v[1] for v in accuracies[task].values())
        mean_acc = sum(v[0] * v[1] for v in accuracies[task].values()) / task_size
        accuracies[task]["mean_acc"] = mean_acc
        print(f"{task}: {mean_acc:.3f}")

    return accuracies


def calc_acc(
    task: str,
    model: LanguageModel,
    corpus: Corpus,
    classes: Tensor,
    final_hidden: Tensor,
    use_full_model_probs: bool,
    wfinal_hidden: Optional[Tensor] = None,
) -> float:
    probs = torch.bmm(model.decoder_w[classes], final_hidden.unsqueeze(2))
    probs = probs[:, :, 0]
    probs += model.decoder_b[classes]

    if "npi" in task:
        wprobs = torch.bmm(model.decoder_w[classes], wfinal_hidden.unsqueeze(2))
        wprobs = wprobs[:, :, 0]
        wprobs += model.decoder_b[classes]

        if use_full_model_probs:
            ever_idx = corpus.vocab.stoi["ever"]
            probs = log_softmax(probs, dim=1)[:, ever_idx]
            wprobs = log_softmax(wprobs, dim=1)[:, ever_idx]

        acc = torch.mean((probs >= wprobs).to(torch.float)).item()
    else:
        acc = torch.mean((probs[:, 0] >= probs[:, 1]).to(torch.float)).item()

    return acc
