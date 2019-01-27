import json
from argparse import ArgumentParser
from pprint import pprint
from typing import Set

from ..typedefs.config import ArgDict, ConfigDict


class ExtractConfig:
    """ Sets up the configuration for extraction.

    Config can be provided from a json file or the commandline. Values
    in the json file can be overwritten by providing them from the
    commandline. Will raise an error if a required argument is not
    provided in either the json file or as a commandline arg.

    Attributes
    ----------
    required_args : Set[str]
        All args that should be present in the json file or as a commandline arg.
    parser : ArgumentParser
        argparser that reads in the provided arguments
    config_dict : ConfigDict
        Dicitonary containing separate ArgDicts that are used for extraction.
    """
    def __init__(self) -> None:
        self.required_args: Set[str] = \
            {'model', 'vocab', 'corpus', 'lm_module', 'activation_names', 'output_dir'}

        self.parser = self._init_argparser()
        self.config_dict = self._load_config()

    @staticmethod
    def _init_argparser() -> ArgumentParser:
        parser = ArgumentParser()

        # create group to load config from a file
        from_config = parser.add_argument_group('From config file',
                                                'Provide full experiment setup via config file')
        from_config.add_argument('-c', '--config',
                                 help='Location of json file containing extraction config.')

        # create group to provide info via commandline arguments
        from_cmd = parser.add_argument_group('From commandline',
                                             'Specify experiment setup via commandline arguments')
        from_cmd.add_argument('--model',
                              help='Location of model parameters')
        from_cmd.add_argument('--vocab',
                              help='Location of model vocabulary')
        from_cmd.add_argument('--corpus',
                              help='Location of labeled corpus')
        from_cmd.add_argument('--lm_module',
                              help='Folder containing model module')
        from_cmd.add_argument('--activation_names',
                              help='Activations to be extracted', nargs='*')
        from_cmd.add_argument('--output_dir',
                              help='Path to which extracted embeddings will be written.')
        from_cmd.add_argument('--init_lstm_states_path',
                              help='Location of initial lstm states of the model')
        from_cmd.add_argument('--print_every', type=int,
                              help='Print extraction progress every n steps')
        from_cmd.add_argument('--cutoff', type=int,
                              help='Stop extraction after n sentences. '
                                   'Defaults to -1 to extract entire corpus')

        return parser

    def _load_config(self) -> ConfigDict:
        init_arg_dict = vars(self.parser.parse_args())

        json_provided: bool = init_arg_dict['config'] is not None

        # Load arguments from config
        arg_dict: ArgDict = {}
        if json_provided:
            arg_dict: ArgDict = self._load_config_from_file(init_arg_dict['config'])

        self._validate_config(arg_dict, init_arg_dict)

        self._overwrite_config_json(arg_dict, init_arg_dict, json_provided)

        self._cast_activation_names(arg_dict)

        self._pprint_arg_dict(arg_dict)

        return self._create_config_dict(arg_dict)

    @staticmethod
    def _load_config_from_file(filename: str) -> ArgDict:
        print(f'Loading config setup provided in {filename}')
        with open(filename) as f:
            arg_dict = json.load(f)
        return arg_dict

    def _validate_config(self, arg_dict: ArgDict, init_arg_dict: ArgDict) -> None:
        """ Check if required args are provided """
        for arg in self.required_args:
            arg_present = arg in arg_dict.keys() or init_arg_dict[arg] is not None
            assert arg_present, self.parser.error(f'--{arg} should be provided')

    @staticmethod
    def _overwrite_config_json(arg_dict: ArgDict,
                               init_arg_dict: ArgDict,
                               json_provided: bool) -> None:
        """ Overwrite provided config values with commandline args """
        for arg, val in init_arg_dict.items():
            if val is not None and arg != 'config':
                arg_dict[arg] = val
                if json_provided:
                    print(f'Overwriting {arg} value that was provided in config json')

    @staticmethod
    def _cast_activation_names(arg_dict: ArgDict) -> None:
        """ Cast activation names to (layer, name) format """
        for i, name in enumerate(arg_dict['activation_names']):
            arg_dict['activation_names'][i] = int(name[-1]), name[0:-1]

    @staticmethod
    def _pprint_arg_dict(arg_dict: ArgDict) -> None:
        print()
        pprint(arg_dict)
        print()

    def _create_config_dict(self, arg_dict: ArgDict) -> ConfigDict:
        provided_args = arg_dict.keys()
        init_args = self.required_args
        extract_args = {'cutoff', 'print_every'}

        config_dict = {
            'init': {k: arg_dict[k] for k in init_args & provided_args},
            'extract': {k: arg_dict[k] for k, v in extract_args & provided_args},
        }

        return config_dict
