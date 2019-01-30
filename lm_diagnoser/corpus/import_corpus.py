import pickle
from typing import Any, Dict, List

from typedefs.corpus import LabeledCorpus, LabeledSentence


def convert_to_labeled_corpus(corpus_path: str) -> LabeledCorpus:
    labeled_corpus = {}

    with open(corpus_path, 'rb') as f:
        corpus: Dict[int, Dict[str, Any]] = pickle.load(f)

    for key, item in corpus.items():
        assert 'sen' in item.keys() and 'labels' in item.keys(), 'Corpus item has wrong format.'

        sen = item['sen']
        labels = item['labels']
        misc_info = {k: v for k, v in item.items() if k not in ['sen', 'labels']}
        labeled_sentence = LabeledSentence(sen, labels, misc_info)
        labeled_sentence.validate()
        labeled_corpus[key] = labeled_sentence

    return labeled_corpus


def read_gulordava_corpus(corpus_dir: str) -> LabeledCorpus:
    """
    Read the corpus generated by [1] with 41 sentences, of which all content words were randomly replaced with another
    word of the same form nine time, resulting in 410 sentences in total.

    [1] https://arxiv.org/pdf/1803.11138.pdf

    Parameters
    ----------
    corpus_dir: str
        Directory to corpus files.

    Returns
    -------
    labelled_corpus: dict
        Corpus of labelled sentences as a dictionary from sentence id as key to LabelledSentence object as value.
    """
    def _read_file(path: str) -> List[str]:
        with open(path, "r") as f:
            return f.readlines()

    sentences = _read_file(f"{corpus_dir}/generated.text")
    sentence_info = _read_file(f"{corpus_dir}/generated.tab")[1:]  # Skip header line
    labelled_corpus = {}

    for i, sentence in enumerate(sentences):
        right_info, wrong_info = sentence_info[2*i], sentence_info[2*i+1]

        # Parse lines
        right_info, wrong_info = right_info.split("\t"), wrong_info.split("\t")
        constr_id, sent_id, correct_number, right_form, class_, type_ = right_info[1:7]
        len_context, len_prefix, sent = right_info[11:14]
        constr_id_wrong, sent_id_wrong, _, wrong_form, class_wrong, type_wrong = wrong_info[1:7]
        sent_wrong = wrong_info[13]

        assert class_ == "correct" and class_wrong == "wrong"
        assert constr_id == constr_id_wrong and sent_id == sent_id_wrong and sent == sent_wrong and type_ == type_wrong

        len_prefix, len_context = int(len_prefix), int(len_context)
        subj_pos = len_prefix - len_context
        verb_pos = len_prefix
        sentence = sent.split()

        misc_info = {
            "raw": sent,
            "subj_pos": subj_pos,
            "verb_pos": verb_pos,
            "right_form": right_form,
            "wrong_form": wrong_form,
            "correct_number": correct_number,
            "sent_id": sent_id,
            "constr_id": constr_id,
            "type": type_
        }

        labelled_sentence = LabeledSentence(
            sentence, labels=[0 if correct_number == "sing" else 1] * len(sentence), misc_info=misc_info
        )
        labelled_sentence.validate()
        labelled_corpus[i] = labelled_sentence

    return labelled_corpus
