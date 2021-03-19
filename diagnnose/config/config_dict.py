import json
from argparse import ArgumentParser
from datetime import datetime
from functools import reduce
from pprint import pprint
from typing import Any, Dict, List

from diagnnose.utils.misc import merge_dicts

from .arg_descriptions import arg_descriptions

# group -> arg_name -> value
ConfigDict = Dict[str, Dict[str, Any]]


def create_config_dict() -> ConfigDict:
    """Sets up the configuration for extraction.

    Config can be provided from a json file or the commandline. Values
    in the json file can be overwritten by providing them from the
    commandline. Will raise an error if a required argument is not
    provided in either the json file or as a commandline arg.

    Commandline args should be provided as dot-separated values, where
    the first dot indicates the arg group the arg belongs to.
    For example, setting the ``state_dict`` of a ``model`` can be done
    with the flag ``--model.state_dict state_dict``.

    Returns
    -------
    config_dict : ConfigDict
        Dictionary mapping each arg group to their config values.
    """
    arg_parser = _create_arg_parser()

    args, unk_args = arg_parser.parse_known_args()
    cmd_args = vars(args)

    # Load arguments from config
    init_config_dict: ConfigDict = {}
    if cmd_args["config"] is not None:
        with open(cmd_args.pop("config")) as f:
            init_config_dict.update(json.load(f))

    _add_unk_args(cmd_args, unk_args)
    config_dict = _add_cmd_args(init_config_dict, cmd_args)

    _cast_activation_names(config_dict)

    _set_tokenizer(config_dict)

    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    pprint(config_dict)

    return config_dict


def _create_arg_parser() -> ArgumentParser:
    parser = ArgumentParser()

    # create group to load config from a file
    from_config = parser.add_argument_group(
        "From config file", "Provide full experiment setup via config json file."
    )
    from_config.add_argument(
        "-c", "--config", help="Path to json file containing extraction config."
    )

    # create group to provide info via commandline arguments
    # Required args are not set to be required here as they can come from --config
    from_cmd = parser.add_argument_group(
        "From commandline",
        "Specify experiment setup via commandline arguments. "
        "Can be combined with the json config, in which case these"
        " cmd arguments overwrite the config args.",
    )

    for group, group_args in arg_descriptions.items():
        for arg, arg_config in group_args.items():
            arg = f"{group}.{arg}"

            if arg_config.get("type", str) == bool:
                action = (
                    "store_false" if arg_config.get("default", False) else "store_true"
                )
                from_cmd.add_argument(
                    f"--{arg}", action=action, help=arg_config["help"], default=None
                )
            else:
                from_cmd.add_argument(
                    f"--{arg}",
                    nargs=arg_config.get("nargs", None),
                    type=arg_config.get("type", str),
                    help=arg_config["help"],
                )

    return parser


def _add_unk_args(cmd_args: Dict[str, Any], unk_args: List[str]):
    """ Add arguments that are not part of the default arg structure """
    unk_args = [x.split() for x in " ".join(unk_args).split("--") if len(x) > 0]
    for arg in unk_args:
        key = arg[0]
        val = arg[1] if len(arg) == 2 else arg[1:]
        cmd_args[key] = val


def _add_cmd_args(config_dict: ConfigDict, cmd_args: Dict[str, Any]) -> ConfigDict:
    """ Update provided config values with cmd args. """
    cdm_arg_dicts: List[ConfigDict] = []
    for arg, val in cmd_args.items():
        if val is not None:
            cmd_arg_keys = arg.split(".")[::-1]
            cmd_arg_dict = val
            for key in cmd_arg_keys:
                cmd_arg_dict = {key: cmd_arg_dict}
            cdm_arg_dicts.append(cmd_arg_dict)

    if len(cdm_arg_dicts) > 0:
        return reduce(merge_dicts, (config_dict, *cdm_arg_dicts))

    return config_dict


def _cast_activation_names(config_dict: ConfigDict) -> None:
    """Casts activation names tobthe tuple format that is used
    throughout the library.
    """
    # Translate activation names to tuple format that is used in the library
    for group, group_config in config_dict.items():
        if "activation_names" in group_config:
            activation_names = group_config["activation_names"]
            assert all(
                isinstance(a_name, list)
                and isinstance(a_name[0], int)
                and isinstance(a_name[1], str)
                for a_name in activation_names
            ), "Incorrect format for activation names, should be [[layer, name]]."

            config_dict[group]["activation_names"] = list(map(tuple, activation_names))


def _set_tokenizer(config_dict: ConfigDict) -> None:
    """ Set tokenizer name manually for Huggingface models. """
    if (
        "transformer_type" in config_dict.get("model", {})
        and "tokenizer" not in config_dict
    ):
        config_dict["tokenizer"] = {"path": config_dict["model"]["transformer_type"]}

        if "cache_dir" in config_dict["model"]:
            config_dict["tokenizer"]["cache_dir"] = config_dict["model"]["cache_dir"]
