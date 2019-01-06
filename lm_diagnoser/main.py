from argparse import ArgumentParser

from extractors.base_extractor import Extractor

OUTPUT_EMBS_DIR = './embeddings/data/extracted'


def init_argparser() -> ArgumentParser:
    parser = ArgumentParser()

    parser.add_argument('--model', help='Location of json file with model setup')
    parser.add_argument('--corpus', help='Location of labeled corpus')
    parser.add_argument('--activations', default=[(1, 'hx')], help='Activations to be extracted')
    parser.add_argument('--init_embs', default='', help='Location of initial embeddings')
    parser.add_argument('--print_every', type=int, default=10, help='Print progress every n steps')
    parser.add_argument('--cutoff', type=int, default=-1,
                        help='Stop extraction after n sentences. '
                             'Defaults to -1 to extract entire corpus.')
    parser.add_argument('--output_dir', default=OUTPUT_EMBS_DIR,
                        help='Path to which extracted embeddings will be written.')

    return parser


if __name__ == '__main__':
    config = init_argparser().parse_args()

    extractor = Extractor(
        config.model,
        config.corpus,
        config.activations,
    )
    extractor.extract(config.output_dir, print_every=config.print_every, cutoff=config.cutoff)
