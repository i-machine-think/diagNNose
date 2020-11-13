import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

# sen_id -> (licensor, scope, npi_present) -> item
ItemCondition = Tuple[int, int, int]
CorpusItem = Dict[ItemCondition, Dict[str, Any]]
CorpusDict = Dict[int, CorpusItem]

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


def preproc_warstadt(path: str) -> CorpusDict:
    """Reads and preprocesses the NPI corpus of Warstadt et al. (2019).

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
        lines = [line[:-1].split("\t") for line in f]

    def preproc(s):
        return int(s) if s.isnumeric() else s

    # Separate punctuation from adjacent tokens
    pattern = re.compile(r"[.,?]")

    def preproc_sen(s):
        return pattern.sub(lambda m: f" {m.group(0)}", s).split()

    # map each line to a dictionary
    raw_items: List[Dict[str, Any]] = [
        {
            **{k: preproc(v) for k, v in [x.split("=") for x in line[0].split("-")]},
            "correct": bool(int(line[1])),
            "sen": preproc_sen(line[-1]),
        }
        for line in lines
    ]

    extra_idx = 0

    for idx, item in enumerate(raw_items):
        # Remove apostrophes that wrap the full sentence.
        if item["sen"][0][0] == '"':
            item["sen"][0] = item["sen"][0][1:]
        if item["sen"][-1][-1] == '"':
            item["sen"][-1] = item["sen"][-1][:-1]

        sen_id = idx // 8
        # There exist only 4 instead of 8 conditions for `simplequestions`.
        if item["env"] == "simplequestions":
            extra_idx += idx % 8 == 4
        item["sen_id"] = sen_id + extra_idx

        # Cut multi-word NPIs into pieces.
        if item["npi_present"] == 1:
            if item["npi"] in ["atall", "inyears"]:
                item["npi"] = f"{item['npi'][:2]} {item['npi'][2:]}"

    sen_id2items = defaultdict(dict)

    for item in raw_items:
        sen_id2items[item["sen_id"]][
            item["licensor"], item["scope"], item["npi_present"]
        ] = item

    return sen_id2items


def create_downstream_corpus(
    orig_corpus: Union[str, CorpusDict],
    output_path: Optional[str] = None,
    conditions: Optional[List[Tuple[int, int, int]]] = None,
    envs: Optional[List[str]] = None,
    skip_duplicate_items: bool = False,
) -> List[str]:
    """Create a new corpus from the original one that contains the
    subsentences up to the position of the NPI.

    Parameters
    ----------
    orig_corpus : str | CorpusDict
        Either the path to the original corpus, or a CorpusDict that
        has been created using `preproc_warstadt`.
    output_path : str, optional
        Path to the output file that will be created in .tsv format.
        If not provided the corpus won't be written to disk.
    conditions : List[Tuple[int, int, int]], optional
        List of corpus item conditions (licensor, scope, npi_present).
        If not provided the correct NPI cases (1, 1, 1) will be used.
    envs : List[str], optional
        List of of licensing environments that should be used.
    skip_duplicate_items : bool
        Some corpus items only differ in their post-NPI content, and
        will lead to equivalent results on a downstream task. Defaults
        to False.

    Returns
    -------
    corpus : List[str]
        List of strings representing each corpus item. Note that the
        first line of the list contains the .tsv header.
    """
    if envs is None:
        envs = ENVS
    if conditions is None:
        conditions = [(1, 1, 1)]

    if isinstance(orig_corpus, str):
        id2items = preproc_warstadt(orig_corpus)
    else:
        id2items = orig_corpus

    sens_seen = set()
    corpus = [
        "\t".join(
            [
                "orig_sen_idx",
                "condition",
                "sen",
                "counter_sen",
                "token",
                "full_npi",
                "env",
                "labels",
            ]
        )
    ]

    for idx, items in id2items.items():
        if items[1, 1, 1]["env"] not in envs:
            continue

        for condition in conditions:
            licensor, scope, npi_present = condition
            item = items[condition]

            # Flip licensor bool to create an item of the opposite licensing polarity.
            counter_condition = (int(not licensor), scope, npi_present)
            counter_item = items[counter_condition]

            full_npi = item["npi"]
            # For multi-phrase NPIs (at all, in years) we are interested in the final token
            true_npi = full_npi.split()[-1]
            sen = " ".join(item["sen"])
            counter_sen = " ".join(counter_item["sen"])

            # Index of start of NPI phrase, to which we add the index of the final token
            start_idx = sen.index(f" {full_npi} ") + 1
            npi_idx = start_idx + sen[start_idx:].index(true_npi)
            start_idx = counter_sen.index(f" {full_npi} ") + 1
            counter_npi_idx = start_idx + counter_sen[start_idx:].index(true_npi)

            monotonicity = "downward" if licensor == 1 else "upward"

            if sen[:npi_idx] + full_npi in sens_seen and skip_duplicate_items:
                continue

            sens_seen.add(sen[:npi_idx] + full_npi)

            corpus.append(
                "\t".join(
                    (
                        str(idx),
                        str(condition),
                        sen[:npi_idx],
                        counter_sen[:counter_npi_idx],
                        true_npi,
                        full_npi,
                        item["env"],
                        monotonicity,
                    )
                )
            )

    if output_path is not None:
        with open(output_path, "w") as f:
            f.write("\n".join(corpus))

    return corpus


if __name__ == "__main__":
    new_corpus = create_downstream_corpus(
        "../../../lm_data/corpora/downstream/warstadt/npi_data_all_environments.tsv",
        output_path="../../../lm_data/corpora/npi/lc_detection_binary_NEW.tsv",
        conditions=[(1, 1, 1), (0, 1, 1)],
        skip_duplicate_items=True,
    )
    print(len(new_corpus))
