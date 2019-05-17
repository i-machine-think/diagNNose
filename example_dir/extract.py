from diagnnose.config.arg_parser import create_arg_parser
from diagnnose.config.setup import ConfigSetup
from diagnnose.corpora.import_corpus import import_corpus_from_path
from diagnnose.extractors.base_extractor import Extractor
from diagnnose.models.import_model import import_model_from_json
from diagnnose.models.language_model import LanguageModel
from diagnnose.typedefs.corpus import Corpus

if __name__ == '__main__':
    arg_groups = {'model', 'activations', 'corpus', 'extract'}
    arg_parser, required_args = create_arg_parser(arg_groups)

    config_dict = ConfigSetup(arg_parser, required_args, arg_groups).config_dict

    model: LanguageModel = import_model_from_json(config_dict['model'])
    corpus: Corpus = import_corpus_from_path(**config_dict['corpus'])

    extractor = Extractor(model, corpus, **config_dict['activations'])
    extractor.extract(**config_dict['extract'])
