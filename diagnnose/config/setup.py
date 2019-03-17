import json
from argparse import ArgumentParser
from pprint import pprint
from typing import Dict, Set

from ..typedefs.config import ArgDict, ConfigDict


class ConfigSetup:
    """ Sets up the configuration for extraction.

    Config can be provided from a json file or the commandline. Values
    in the json file can be overwritten by providing them from the
    commandline. Will raise an error if a required argument is not
    provided in either the json file or as a commandline arg.

    Parameters
    ----------
    argparser : ArgumentParser
        argparser that reads in the provided arguments
    required_args : Set[str]
        Set of arguments that should be at least provided in either the
        config file or as commandline argument.
    arg_groups : Dict[str, Set[str]]
        Dictionary that maps various groups to the arguments that belong
        to that group. This makes it easier to quickly pass different
        arguments to different parts of a module.

    Attributes
    ----------
    config_dict : ConfigDict
        Dicitonary containing separate ArgDicts that are used for extraction.
    """
    def __init__(self,
                 argparser: ArgumentParser,
                 required_args: Set[str],
                 arg_groups: Dict[str, Set[str]]) -> None:

        self.argparser = argparser
        self.required_args = required_args
        self.arg_groups = arg_groups

        self.config_dict = self._load_config()

    def _load_config(self) -> ConfigDict:
        init_arg_dict = vars(self.argparser.parse_args())

        json_provided: bool = init_arg_dict['config'] is not None

        # Load arguments from config
        arg_dict: ArgDict = {}
        if json_provided:
            self._load_config_from_file(init_arg_dict['config'], arg_dict)

        self._validate_config(arg_dict, init_arg_dict)

        self._overwrite_config_json(arg_dict, init_arg_dict, json_provided)

        if 'activation_names' in arg_dict:
            self._cast_activation_names(arg_dict)

        self._pprint_arg_dict(arg_dict)

        return self._create_config_dict(arg_dict)

    @staticmethod
    def _load_config_from_file(filename: str, arg_dict: ArgDict) -> None:
        print(f'Loading config setup provided in {filename}')
        with open(filename) as f:
            arg_dict.update(json.load(f))

    def _validate_config(self, arg_dict: ArgDict, init_arg_dict: ArgDict) -> None:
        """ Check if required args are provided """
        for arg in self.required_args:
            arg_present = arg in arg_dict.keys() or init_arg_dict.get(arg, None) is not None
            assert arg_present, self.argparser.error(f'--{arg} should be provided')

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
        if 'activation_names' in arg_dict:
            for i, name in enumerate(arg_dict['activation_names']):
                arg_dict['activation_names'][i] = int(name[-1]), name[0:-1]

    @staticmethod
    def _pprint_arg_dict(arg_dict: ArgDict) -> None:
        print()
        pprint(arg_dict)
        print()

    def _create_config_dict(self, arg_dict: ArgDict) -> ConfigDict:
        provided_args = arg_dict.keys()

        config_dict = {}
        for group, args in self.arg_groups.items():
            config_dict[group] = {k: arg_dict[k] for k in args & provided_args}

        return config_dict
