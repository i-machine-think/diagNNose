from argparse import ArgumentParser
import pickle
from typing import Any, List

import numpy as np

from models import import_model_from_json, InterventionLSTM
from corpus import convert_to_labeled_corpus
from interventions import LanguageModelInterventionMechanism
from classifiers import DiagnosticClassifier
from embeddings import InitEmbs
from typedefs.models import PartialActivationDict


def main():
    # Manage config
    argparser = init_argparser()
    config = argparser.parse_args()
    # TODO: Validate parser args

    # Load model and corpus
    model = import_model_from_json(config.model, model_class=InterventionLSTM)
    corpus = convert_to_labeled_corpus(config.corpus)
    step_size = 0.1  # TODO: Make arg
    classifiers = {path: DiagnosticClassifier.load(path) for path in config.classifiers}
    mechanism = LanguageModelInterventionMechanism(model, classifiers, step_size)
    model = mechanism.apply()
    embeddings = InitEmbs(config.embeddings, model)

    run_interventions(model, corpus, embeddings)


def run_interventions(model, corpus, init_embs):

    for sentence_id, labelled_sentence in corpus.items():
        sentence = labelled_sentence.sen
        labels = labelled_sentence.labels

        activations = init_embs.activations

        for token, label in zip(sentence, labels):
            out, activations = model.forward(token, activations, label=label)


def read_labels(path) -> Any:
    with open(path, 'rb') as f:
        labels = pickle.load(f)

    return labels


def init_argparser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument('--model', type=str, help='Location of json file with model setup')
    parser.add_argument('--corpus', type=str, help='Location of labeled corpus')
    parser.add_argument('--classifiers', nargs="+", help='Location of diagnostic classifiers')
    parser.add_argument('--embeddings', type=str, help="Location of initial embeddings.")

    return parser


if __name__ == "__main__":
    main()
