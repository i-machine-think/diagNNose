from argparse import ArgumentParser

import lm_diagnoser
from lm_diagnoser.classifiers.diagnostic_classifier import DiagnosticClassifier
from lm_diagnoser.extractors.base_extractor import Extractor


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
    config['activation_names'] = [tuple(x) for x in config['activation_names'] if x[1][0] in 'io']

    extractor = Extractor(**config)
    extractor.extract()
