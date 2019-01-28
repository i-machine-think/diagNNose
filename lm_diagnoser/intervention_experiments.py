"""
Run an LSTM language model with interventions.
"""

from argparse import ArgumentParser
from typing import Tuple
from copy import deepcopy

import numpy as np
import torch
from torch.nn.functional import softmax, log_softmax

from models import import_model_from_json, InterventionLSTM
from corpus import convert_to_labeled_corpus, read_linzen_corpus, read_gulordava_corpus
from interventions import LanguageModelInterventionMechanism
from classifiers import DiagnosticClassifier
from embeddings import InitEmbs
from typedefs import LabeledCorpus


def main():
    # Manage config
    argparser = init_argparser()
    config = argparser.parse_args()

    # Load data: Corpus, model, diagnostic classifiers
    if config.gulordava:
        corpus = read_gulordava_corpus(config.corpus)
    else:
        corpus = convert_to_labeled_corpus(config.corpus)

    model = import_model_from_json(config.model, model_class=InterventionLSTM)
    step_size = config.step_size
    classifiers = {path: DiagnosticClassifier.load(path) for path in config.classifiers}
    embeddings = InitEmbs(config.embeddings, model)

    # 1. Experiment: Replicate Gulordava findings
    # In what percentage of cases does the LM assign a higher probability to the grammatically correct sentence?
    replicate_gulordava(model, corpus, init_embs=embeddings)

    # Add mechanism to model
    mechanism = LanguageModelInterventionMechanism(model, classifiers, step_size)
    intervention_model = mechanism.apply()


def replicate_gulordava(model: InterventionLSTM,
                        corpora: Tuple[LabeledCorpus, LabeledCorpus],
                        init_embs: InitEmbs) -> float:
    """
    Replicate the Language Model number prediction accuracy experiment from [1]. In this experiment, a language model
    is facing a sentence in which the main verb is presented in its singular and plural form, one of which is
    ungrammatical given the numerosity of the sentence's subject. The LM is then expected to assign a higher probability
    to the grammatical sentence. Finally, the percentage of cases in which this was the case is reported.

    [1] https://arxiv.org/pdf/1803.11138.pdf
    """
    def _get_number_prediction_accuracy(model, corpus, init_embs):
        correct_predictions = 0

        for i, (sentence_id, labelled_sentence) in enumerate(corpus.items()):

            if i % 20 == 0:
                print(i)

            # Get necessary information
            sentence = labelled_sentence.sen
            target_pos = labelled_sentence.misc_info["verb_pos"]
            right_form = labelled_sentence.misc_info["right_form"]
            wrong_form = labelled_sentence.misc_info["wrong_form"]

            # Retrieve word indices
            right_index = model.w2i[right_form]
            wrong_index = model.w2i[wrong_form]

            # Feed sentence into RNN
            activations = init_embs.activations

            for pos, token in enumerate(sentence):
                out, activations = model.forward(token, activations)

                # After processing the sentence up to the verb in question, check which of the verb forms is assigned
                # a higher probability
                if pos == target_pos - 1:
                    if out[right_index] > out[wrong_index]:
                        correct_predictions += 1
                    break

        num_sentences = len(corpus)
        acc = correct_predictions / num_sentences
        return acc

    # Calculate scores
    labelled_original_corpus, labelled_nonce_corpus = corpora
    original_acc = _get_number_prediction_accuracy(model, labelled_original_corpus, init_embs)
    nonce_acc = _get_number_prediction_accuracy(model, labelled_nonce_corpus, init_embs)

    print(f"Original accuracy: {original_acc * 100:.1f}")
    print(f"Nonce accuracy: {nonce_acc * 100:.1f}")


def init_argparser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument('--model', type=str, help='Location of json file with model setup')
    parser.add_argument('--corpus', type=str, help='Location of corpus')
    parser.add_argument('--gulordava', action="store_true", help="Flag to indicate whether the Gulordava corpus is used.")
    parser.add_argument('--classifiers', nargs="+", help='Location of diagnostic classifiers')
    parser.add_argument('--embeddings', type=str, help="Location of initial embeddings.")
    parser.add_argument('--step_size', type=float, default=0.1, help="Step-size for weakly supervised interventions.")

    return parser


if __name__ == "__main__":
    main()
