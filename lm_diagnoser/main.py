from models.forward_lstm import ForwardLSTM
from extractors.base_extractor import Extractor
# from classifiers.logreg import run_experiments
from customtypes.corpus import LabeledCorpus, LabeledSentence, Labels


MODEL_DIR = './models'
OUTPUT_EMBS_DIR = './embeddings/data/extracted'
PRETRAINED_EMBS_DIR = './embeddings/data/init'
LABELED_DATA_DIR = './corpus/data/labeled'

models = {
    'gulordava': ForwardLSTM(
        'model.pt',
        'vocab.txt',
        MODEL_DIR + '/gulordava/'
    ),
}

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
