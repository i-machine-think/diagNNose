from argparse import ArgumentParser

from classifiers.diagnostic_classifier import DiagnosticClassifier
from extractors.base_extractor import Extractor


def init_argparser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument('-e', '--extraction_setup', required=True,
                        help='Location of json file containing extraction config.')
    # parser.add_argument('--model', help='Location of json file with model setup')
    # parser.add_argument('--corpus', help='Location of labeled corpus')
    # parser.add_argument('--activations', help='Activations to be extracted')
    # parser.add_argument('--init_embs', help='Location of initial embeddings')
    # parser.add_argument('--print_every', type=int, help='Print extraction progress every n steps')
    # parser.add_argument('--cutoff', type=int, help='Stop extraction after n sentences. '
    #                     'Defaults to -1 to extract entire corpus.')
    # parser.add_argument('--output_dir', help='Path to which extracted embeddings will be written.')

    return parser


# TODO: allow extractor arguments both directly or from json config
# TODO: Make extraction/classification optional from argparser
if __name__ == '__main__':
    config = init_argparser().parse_args()
    extraction_setup = config.extraction_setup

    import json
    with open(extraction_setup) as f:
        config = json.load(f)
    print(config)

    # TODO: HOTFIX
    config['activations'] = [tuple(x) for x in config['activations']]

    extractor = Extractor(config)
    extractor.extract()

    classifier = DiagnosticClassifier(
        config['output_dir'],
        config['activations'],
        650
    )
    classifier.classify()
