from collections import namedtuple

from models.lstm import Forward_LSTM
from extractors.init_extractor import extract
from classifiers.logreg import run_experiments


MODEL_DIR = './models/'
PRETRAINED_EMBS_DIR = './data/pretrained_embs/'
PARSED_DATA_DIR = './data/parsed/'

GapSentence = namedtuple('GapSentence', ['sen', 'raw', 'labels', 'filler', 'gap_start', 'gap_end', 'dep_len'])
LM = namedtuple('LM', ['model_type', 'model_path', 'vocab_path', 'init_embs', 'vocab_size', 'hidden_size'])

models = {
    'gulordava': Forward_LSTM(
        50001,
        650,
        650,
        50001,
        MODEL_DIR+'gulordava/vocab.txt',
        MODEL_DIR+'gulordava/model.pt',
    ),
}

data_config = {
    'init_embs': PRETRAINED_EMBS_DIR+'avgs.pickle',
    'parsed_data': PARSED_DATA_DIR+'gapsens.pickle',
}

if __name__ == '__main__':
    extract(models['gulordava'], data_config, cutoff=1000)
    # run_experiments('data/extracted_embs/hx-50000_l0.pickle', 'data/extracted_embs/50000-labels.pickle', save_model=False)
