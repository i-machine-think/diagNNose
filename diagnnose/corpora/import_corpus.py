import os
from typing import Any, Dict, List

from rnnalyse.typedefs.corpus import Corpus, CorpusSentence
from rnnalyse.utils.paths import load_pickle


def import_corpus_from_path(corpus_path: str, from_dict: bool = False) -> Corpus:
    labeled_corpus = {}

    if from_dict:
        init_corpus: Dict[int, Dict[str, Any]] = load_pickle(os.path.expanduser(corpus_path))
    else:
        init_corpus = read_raw_corpus(corpus_path)

    for key, item in init_corpus.items():
        assert 'sen' in item.keys() or 'sent' in item.keys(), \
            'Corpus item should contain a sentence (\'sen\' or \'sent\') attribute'

        sen = item['sent'] if 'sent' in item else item['sen']
        labels = item['labels'] if 'labels' in item else None
        misc_info = {k: v for k, v in item.items() if k not in ['sen', 'sent', 'labels']}
        labeled_sentence = CorpusSentence(sen, labels, misc_info)
        labeled_sentence.validate()
        labeled_corpus[key] = labeled_sentence

    return labeled_corpus


def read_raw_corpus(corpus_path: str, separator: str = '\t') -> Dict[int, Dict[str, Any]]:
    with open(corpus_path) as f:
        lines = f.read().strip().split('\n')

    split_lines = [l.strip().split(separator) for l in lines]
    header = split_lines[0]
    corpus_lines = split_lines[1:]

    init_corpus = {
        i: string_to_dict(header, x) for i, x in enumerate(corpus_lines)
    }

    return init_corpus


def string_to_dict(header: List[str], line: List[str]) -> Dict[str, Any]:
    sendict: Dict[str, Any] = dict(zip(header, line))
    if 'sen' in sendict:
        sendict['sen'] = sendict['sen'].strip().split(' ')
    if 'sent' in sendict:
        sendict['sent'] = sendict['sent'].strip().split(' ')

    return sendict
