from lm_diagnoser.config.extract import ExtractConfig
from lm_diagnoser.corpora.import_corpus import convert_to_labeled_corpus
from lm_diagnoser.extractors.base_extractor import Extractor
from lm_diagnoser.models.import_model import import_model_from_json
from lm_diagnoser.models.language_model import LanguageModel
from lm_diagnoser.typedefs.config import ConfigDict
from lm_diagnoser.typedefs.corpus import LabeledCorpus

if __name__ == '__main__':
    config_dict: ConfigDict = ExtractConfig().config_dict

    model: LanguageModel = import_model_from_json(**config_dict['model'])
    corpus: LabeledCorpus = convert_to_labeled_corpus(**config_dict['corpus'])

    extractor = Extractor(model, corpus, **config_dict['init_extract'])
    extractor.extract(**config_dict['extract'])
