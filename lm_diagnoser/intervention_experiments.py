"""
Run an LSTM language model with interventions.
"""

from argparse import ArgumentParser

from classifiers.diagnostic_classifier import DiagnosticClassifier
from corpus import convert_to_labeled_corpus, read_gulordava_corpus
from embeddings.eos import create_average_eos_representations
from embeddings.initial import InitEmbs
from interventions.weakly_supervised import LanguageModelInterventionMechanism
from models.intervention_lstm import InterventionLSTM
from models.import_model import import_model_from_json
from typedefs.corpus import LabeledCorpus


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
    #create_average_eos_representations(model, corpus, "embeddings/data/extracted/gulordava/eos.pickle")

    # 1. Experiment: Replicate Gulordava findings
    # In what percentage of cases does the LM assign a higher probability to the grammatically correct sentence?
    replicate_gulordava(model, corpus, init_embs=embeddings)

    # Add mechanism to model
    #mechanism = LanguageModelInterventionMechanism(model, classifiers, step_size)
    #intervention_model = mechanism.apply()


def replicate_gulordava(model: InterventionLSTM,
                        corpus: LabeledCorpus,
                        init_embs: InitEmbs) -> None:
    """
    Replicate the Language Model number prediction accuracy experiment from [1]. In this experiment, a language model
    is facing a sentence in which the main verb is presented in its singular and plural form, one of which is
    ungrammatical given the numerosity of the sentence's subject. The LM is then expected to assign a higher probability
    to the grammatical sentence. Finally, the percentage of cases in which this was the case is reported.

    [1] https://arxiv.org/pdf/1803.11138.pdf
    """
    print("Replicating Gulordava Number Agreement experiment...")

    # Calculate scores
    scores = {"original": [], "generated": []}

    for i, (sentence_id, labelled_sentence) in enumerate(corpus.items()):

        if i % 5 == 0:
            print(f"\rProcessing sentence #{i+1}...", end="", flush=True)

        # Get necessary information
        sentence = labelled_sentence.sen
        target_pos = labelled_sentence.misc_info["verb_pos"]
        right_form = labelled_sentence.misc_info["right_form"]
        wrong_form = labelled_sentence.misc_info["wrong_form"]
        sentence_type = labelled_sentence.misc_info["type"]

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
                    scores[sentence_type].append(1)
                else:
                    scores[sentence_type].append(0)

    original_acc = sum(scores["original"]) / len(scores["original"])
    nonce_acc = sum(scores["generated"]) / len(scores["generated"])

    print("")
    print(f"Original accuracy: {round(original_acc * 100, 1):.1f}")
    print(f"Nonce accuracy: {round(nonce_acc * 100, 1):.1f}")


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
