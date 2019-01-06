import argparse

from extractors.base_extractor import Extractor

OUTPUT_EMBS_DIR = './embeddings/data/extracted'
LABELED_DATA_DIR = './corpus/data/labeled'

corpora = {
    'gapsens': LABELED_DATA_DIR + '/gapsens.pickle'
}


if __name__ == '__main__':
    extractor = Extractor(
        models['gulordava'],
        corpora['gapsens'],
        [(1, 'hx'), (1, 'cx')]
    )
    extractor.extract(OUTPUT_EMBS_DIR, print_every=5)
