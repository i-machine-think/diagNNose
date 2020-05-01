from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

# sen_id -> (licensor, scope, npi_present) -> item
CorpusDict = Dict[int, Dict[Tuple[int, int, int], Dict[str, Any]]]
EnvIdDict = Dict[str, List[int]]

ENVS = [
    "adverbs",
    "conditional",
    "determiner_negation_biclausal",
    "only",
    "quantifier",
    "questions",
    "sentential_negation_biclausal",
    "simplequestions",
    "superlative",
]


def preproc_warstadt(path: str) -> Tuple[CorpusDict, EnvIdDict]:
    """ Reads and preprocesses the NPI corpus of Warstadt et al. (2019).

    Paper: https://arxiv.org/pdf/1901.03438.pdf

    Data: https://alexwarstadt.files.wordpress.com/2019/08/npi_lincensing_data.zip

    Parameters
    ----------
    path : str
        Path to .tsv corpus file.

    Returns
    -------
    sen_id2items : CorpusDict
        Dictionary mapping a sen_id to a triplet (licensor, scope,
        npi_present) to the full corpus item.
    env2sen_ids : EnvIdDict
        Dictionary mapping each env type to a list of sen_id's of that
        type.
    """
    with open(path) as f:
        fields = [line[:-1].split("\t") for line in f]

    def preproc(s):
        if s.isnumeric():
            return int(s)
        return s

    # map each line to a dictionary
    sendict: List[Dict[str, Any]] = [
        {
            **{k: preproc(v) for k, v in [x.split("=") for x in l[0].split("-")]},
            "correct": bool(int(l[1])),
            "sen": l[-1]
            .replace(".", " .")
            .replace(",", " ,")
            .replace("?", " ?")
            .split(),
        }
        for l in fields
    ]

    extra_idx = 0

    for i, x in enumerate(sendict):
        # Add spaces to apostrophes that are attached to tokens.
        if x["sen"][0][0] == '"':
            x["sen"][0] = x["sen"][0][1:]
        if x["sen"][-1][-1] == '"':
            x["sen"][-1] = x["sen"][-1][:-1]

        lowsen = list(map(lambda w: w.lower(), x["sen"]))
        sen_id = i // 8
        # There exist only 4 instead of 8 conditions for `simplequestions`.
        if x["env"] == "simplequestions":
            extra_idx += i % 8 == 4
        x["sen_id"] = sen_id + extra_idx

        # Locate the index of the crucial licensing item
        # The try .. except structure fixes minor errors in the original file.
        try:
            if (
                x["crucial_item"] in ["none of the", "more than three"]
                or x["crucial_item"][:4] == "most"
            ):
                x["crucial_idx"] = lowsen.index(x["crucial_item"][:4])
            elif x["crucial_item"] == "a lot of":
                x["crucial_idx"] = lowsen.index("lot") - 1
            else:
                x["crucial_idx"] = lowsen.index(x["crucial_item"])
        except ValueError:
            if x["crucial_item"] == "none of the" and "no" in lowsen:
                x["crucial_item"] = "no"
                x["crucial_idx"] = lowsen.index("no")
            elif x["crucial_item"] == "no" and "none" in lowsen:
                x["crucial_item"] = "none of the"
                x["crucial_idx"] = lowsen.index("none")
            elif x["crucial_item"] == "whether":
                x["crucial_idx"] = None
            else:
                x["crucial_idx"] = None

        # Locates the index of the npi, and its distance to the licensor.
        if x["npi_present"] == 1:
            if x["npi"] in ["atall", "inyears", "at all", "in years"]:
                if " " not in x["npi"]:
                    x["npi"] = f"{x['npi'][:2]} {x['npi'][2:]}"
                npi_idx = None
                for j, (w1, w2) in enumerate(zip(lowsen[:-1], lowsen[1:])):
                    if [w1, w2] == x["npi"].split():
                        npi_idx = j
            else:
                npi_idx = lowsen.index(x["npi"])
            x["npi_idx"] = npi_idx

            if x["crucial_idx"] is not None:
                x["distance"] = npi_idx - x["crucial_idx"]
            else:
                x["distance"] = None

    sen_id2items = defaultdict(dict)

    for x in sendict:
        sen_id2items[x["sen_id"]][x["licensor"], x["scope"], x["npi_present"]] = x

    env2sen_ids: EnvIdDict = {env: [] for env in ENVS}
    for idx, items in sen_id2items.items():
        env = items[1, 1, 1]["env"]
        env2sen_ids[env].append(idx)

    return sen_id2items, env2sen_ids


def create_downstream_corpus(
    orig_corpus: Union[str, CorpusDict],
    output_path: str,
    conditions: Optional[List[Tuple[int, int, int]]] = None,
    envs: Optional[List[str]] = None,
) -> List[str]:
    """ Create a new corpus from the original one that contains the
    subsentences up to the position of the NPI.

    Parameters
    ----------
    orig_corpus : str | CorpusDict
        Either the path to the original corpus, or a CorpusDict that
        has been created using `preproc_warstadt`.
    output_path : str
        Path to the output file that will be created
    conditions : List[Tuple[int, int, int]], optional
        List of corpus item conditions (licensor, scope, npi_present).
        If not provided the correct NPI cases (1, 1, 1) and the cases
        without a licensor (0, 1, 1) will be used.
    envs: List[str], optional
        List of of licensing environments that should be used.

    Returns
    -------
    corpus : List[str]
        List of strings representing each corpus item. The .tsv header
        is not returned.
    """
    if envs is None:
        envs = ENVS

    if isinstance(orig_corpus, str):
        id2items = preproc_warstadt(orig_corpus)[0]
    else:
        id2items = orig_corpus

    sens_seen = set()
    if conditions is None:
        corpus = ["\t".join(["idx", "sen", "npi", "wsen", "env"])]
    else:
        corpus = ["\t".join(["idx", "sen", "npi", "condition", "env", "labels"])]

    for idx, items in id2items.items():
        if items[1, 1, 1]["env"] not in envs:
            continue

        if conditions is None:
            correct_item = items[1, 1, 1]
            wrong_item = items[0, 1, 1]

            npi = correct_item["npi"]
            correct_sen = " ".join(correct_item["sen"])
            wrong_sen = " ".join(wrong_item["sen"])

            correct_npi_idx = correct_sen.index(npi)
            wrong_npi_idx = wrong_sen.index(npi)

            if correct_sen + npi in sens_seen:
                continue

            sens_seen.add(correct_sen + npi)
            corpus.append(
                "\t".join(
                    (
                        str(idx),
                        correct_sen[:correct_npi_idx],
                        npi,
                        wrong_sen[:wrong_npi_idx],
                        correct_item["env"],
                    )
                )
            )
        else:
            for label, condition in enumerate(conditions):
                item = items[condition]

                sen = " ".join(item["sen"])
                npi = item["npi"]
                npi_idx = sen.index(npi)

                corpus.append(
                    "\t".join(
                        (
                            str(idx),
                            sen[:npi_idx],
                            npi,
                            str(condition),
                            item["env"],
                            str(label),
                        )
                    )
                )

    with open(output_path, "w") as f:
        f.write("\n".join(corpus))

    return corpus[1:]


# TODO: combine into 1 with create_downstream_corpus?
def create_lc_detection_corpus(
    orig_corpus: Union[str, CorpusDict],
    output_path: str,
    envs: Optional[List[str]] = None,
    label_each_env: bool = True,
    use_full_sens: bool = True,
) -> List[str]:
    """ Create a new corpus from the original one that contains the
    subsentences up to the position of the NPI, labeled by the
    corresponding environment.

    Parameters
    ----------
    orig_corpus : str
        The path to the original corpus.
    output_path : str
        Path to the output file that will be created
    envs: List[str], optional
        List of of licensing environments that should be used.
    label_each_env : bool, optional
        Label each item with their respective environment (+ polarity),
        or only mark the polarity. Defaults to True.
    use_full_sens : bool, optional
        Toggle to save the full sentence instead of only up till the
        npi position. Defaults to True.

    Returns
    -------
    corpus : List[str]
        List of strings representing each corpus item.
    """
    if envs is None:
        envs = ENVS

    id2items = preproc_warstadt(orig_corpus)[0]

    # We keep track of all subsentences (up to the position of the NPI) that have been seen.
    sens_seen = set()
    corpus = ["sen\tlabels\tenv\tnpi\tnpi_idx\tlc"]

    for items in id2items.values():
        if items[1, 1, 1]["env"] not in envs:
            continue

        correct_item = items[1, 1, 1]
        wrong_item = items[0, 1, 1]

        npi = correct_item["npi"]
        correct_sen = " ".join(correct_item["sen"])
        wrong_sen = " ".join(wrong_item["sen"])

        correct_npi_idx = correct_sen.index(f" {npi} ")
        wrong_npi_idx = wrong_sen.index(f" {npi} ")

        env = correct_item["env"]

        if label_each_env:
            correct_label = correct_item["env"] + "_downward"
            wrong_label = correct_item["env"] + "_upward"
        else:
            correct_label = "downward"
            wrong_label = "upward"

        if use_full_sens:
            wrong_sen = " ".join(items[0, 1, 0]["sen"])
        else:
            correct_sen = correct_sen[:correct_npi_idx]
            wrong_sen = wrong_sen[:wrong_npi_idx]

        # if correct_sen + correct_label + npi in sens_seen:
        #     continue

        sens_seen.add(correct_sen + correct_label + npi)
        corpus.extend(
            [
                "\t".join(
                    (
                        correct_sen,
                        correct_label,
                        env,
                        npi,
                        str(correct_sen[:correct_npi_idx].count(" ") + 1),
                        correct_item["crucial_item"],
                    )
                ),
                "\t".join(
                    (
                        wrong_sen,
                        wrong_label,
                        env,
                        npi,
                        str(wrong_sen[:wrong_npi_idx].count(" ") + 1),
                        wrong_item["crucial_item"],
                    )
                ),
            ]
        )

    with open(output_path, "w") as f:
        f.write("\n".join(corpus))

    return corpus


if __name__ == "__main__":
    new_corpus = create_lc_detection_corpus(
        "../../../lm_data/corpora/downstream/warstadt/npi_data_all_environments.tsv",
        "../../../lm_data/corpora/npi/lc_detection_binary.tsv",
        label_each_env=False,
        use_full_sens=False,
    )
    print(len(new_corpus))
