"""
Run an LSTM language model with interventions.
"""

from argparse import ArgumentParser
from collections import defaultdict

import torch
from torch.nn.functional import log_softmax

from classifiers.diagnostic_classifier import DiagnosticClassifier
from corpus import convert_to_labeled_corpus, read_gulordava_corpus
from embeddings.eos import create_average_eos_representations
from embeddings.initial import InitEmbs
from interventions.weakly_supervised import (
    LanguageModelInterventionMechanism, SubjectLanguageModelInterventionMechanism
)
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

    basic_model = import_model_from_json(config.model, model_class=InterventionLSTM)
    subj_intervention_model = import_model_from_json(config.model, model_class=InterventionLSTM)
    global_intervention_model = import_model_from_json(config.model, model_class=InterventionLSTM)

    # Load classifiers and apply intervention mechanisms
    step_size = config.step_size
    classifiers = {path: DiagnosticClassifier.load(path) for path in config.classifiers}
    subj_mechanism = SubjectLanguageModelInterventionMechanism(subj_intervention_model, classifiers, step_size)
    global_mechanism = LanguageModelInterventionMechanism(global_intervention_model, classifiers, step_size)
    subj_intervention_model = subj_mechanism.apply()
    global_intervention_model = global_mechanism.apply()
    embeddings = InitEmbs(config.embeddings, basic_model)
    #create_average_eos_representations(basic_model, corpus, "embeddings/data/extracted/gulordava/eos.pickle")

    # 1. Experiment: Replicate Gulordava findings
    # In what percentage of cases does the LM assign a higher probability to the grammatically correct sentence?
    #replicate_gulordava(basic_model, corpus, init_embs=embeddings)

    # 2. Experiment: Assess the influence of interventions on LM perplexity
    measure_influence_on_perplexity(basic_model, subj_intervention_model, global_intervention_model, corpus, embeddings)

    # 3. Experiment: Check to what extend the accuracy of Diagnostic Classifiers increases after having interventions
    # on the subject position / on every position
    # TODO

    # 4. Experiment: Repeat the 1. Experiment but measure the influence of interventions on the subject position /
    # on every position
    # TODO


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


def measure_influence_on_perplexity(basic_model: InterventionLSTM,
                                    subj_intervention_model: InterventionLSTM,
                                    global_intervention_model: InterventionLSTM,
                                    corpus: LabeledCorpus,
                                    init_embs: InitEmbs) -> None:
    """
    Check whether interventions - be it only on the subject position or all positions - influence the perplexity
    of the Language Model in a statistically significant way.
    """
    perplexities = defaultdict(list)
    w2i = basic_model.w2i  # Vocabulary is shared between models
    unk_index = basic_model.unk_idx
    basic_activations, subj_activations, global_activations = init_embs.activations, init_embs.activations, init_embs.activations

    print("Assessing influence of interventions on perplexities...")
    print("Gathering perplexity scores for corpus...")

    for sentence_id, sentence in corpus.items():

        if sentence_id % 5 == 0:
            print(f"\rProcessing sentence #{sentence_id+1}...", end="", flush=True)

        basic_perplexity, subj_perplexity, global_perplexity = 0, 0, 0

        # Get necessary sentence data
        sen = sentence.sen
        labels = sentence.labels
        subj_pos = sentence.misc_info["subj_pos"]

        for pos, token in enumerate(sen):
            basic_out, basic_activations = basic_model(token, basic_activations)
            subj_out, subj_activations = subj_intervention_model(
                token, subj_activations, label=labels[pos], is_subj_pos=subj_pos == pos
            )
            global_out, global_activations = global_intervention_model(token, global_activations, label=labels[pos])

            # "Batchify" to speed up expensive log-softmax
            token_index = unk_index if token not in w2i else w2i[token]
            outs = torch.stack((basic_out, subj_out, global_out))
            vocab_probs = log_softmax(outs, dim=0)
            token_probs = vocab_probs[:, token_index]
            basic_prob, subj_prob, global_prob = torch.split(token_probs, 1, dim=0)
            basic_perplexity += basic_prob
            subj_perplexity += subj_prob
            global_perplexity += global_prob

        # Save sentence perplexities
        perplexities["basic"].append(basic_perplexity.detach().numpy()[0])
        perplexities["subj"].append(subj_perplexity.detach().numpy()[0])
        perplexities["global"].append(global_perplexity.detach().numpy()[0])

    # TODO: Test if distribution of perplexities in normally distributed
    # TODO: Test if difference are statistically significant


def init_argparser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument('--model', type=str, help='Location of json file with model setup')
    parser.add_argument('--corpus', type=str, help='Location of corpus')
    parser.add_argument('--gulordava', action="store_true", help="Flag to indicate whether the Gulordava corpus is used.")
    parser.add_argument('--classifiers', nargs="+", help='Location of diagnostic classifiers')
    parser.add_argument('--embeddings', type=str, help="Location of initial embeddings.")
    parser.add_argument('--step_size', type=float, default=0.5, help="Step-size for weakly supervised interventions.")

    return parser


if __name__ == "__main__":
    main()
