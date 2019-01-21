"""
Run an LSTM language model with interventions.
"""

from argparse import ArgumentParser

from models import import_model_from_json, InterventionLSTM
from corpus import convert_to_labeled_corpus
from interventions import LanguageModelInterventionMechanism
from classifiers import DiagnosticClassifier
from embeddings import InitEmbs
from typedefs import LabeledCorpus


def main():
    # Manage config
    argparser = init_argparser()
    config = argparser.parse_args()

    # Load data: Corpus, model, diagnostic classifiers
    model = import_model_from_json(config.model, model_class=InterventionLSTM)
    corpus = convert_to_labeled_corpus(config.corpus)
    step_size = config.step_size
    classifiers = {path: DiagnosticClassifier.load(path) for path in config.classifiers}
    embeddings = InitEmbs(config.embeddings, model)

    # Add mechanism to model
    mechanism = LanguageModelInterventionMechanism(model, classifiers, step_size)
    model = mechanism.apply()

    # Run
    run_interventions(model, corpus, embeddings)


def run_interventions(model: InterventionLSTM,
                      corpus: LabeledCorpus,
                      init_embs: InitEmbs) -> None:
    for sentence_id, labelled_sentence in corpus.items():
        sentence = labelled_sentence.sen
        labels = labelled_sentence.labels
        activations = init_embs.activations

        for token, label in zip(sentence, labels):
            out, activations = model.forward(token, activations, label=label)


def init_argparser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument('--model', type=str, help='Location of json file with model setup')
    parser.add_argument('--corpus', type=str, help='Location of labeled corpus')
    parser.add_argument('--classifiers', nargs="+", help='Location of diagnostic classifiers')
    parser.add_argument('--embeddings', type=str, help="Location of initial embeddings.")
    parser.add_argument('--step_size', type=float, default=0.1, help="Step-size for weakly supervised interventions.")

    return parser


if __name__ == "__main__":
    main()
