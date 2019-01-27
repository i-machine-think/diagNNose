from lm_diagnoser.config.extract import ExtractConfig
from lm_diagnoser.extractors.base_extractor import Extractor
from lm_diagnoser.typedefs.config import ConfigDict

if __name__ == '__main__':
    config_dict: ConfigDict = ExtractConfig().config_dict

    extractor = Extractor(**config_dict['init'])
    extractor.extract(**config_dict['extract'])
