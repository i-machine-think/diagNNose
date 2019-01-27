import json
from argparse import ArgumentParser
from pprint import pprint

from lm_diagnoser.extractors.base_extractor import Extractor


def init_argparser() -> ArgumentParser:
    parser = ArgumentParser()

    # create group to load config from a file
    from_config = parser.add_argument_group('From config file',
                                            'Provide full experiment setup via configuration file')
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


def load_config():
    parser = init_argparser()

    args = parser.parse_args()
    arg_dict = vars(args)

    init_config = {}
    required_args = ['model', 'vocab', 'corpus', 'lm_module', 'activation_names', 'output_dir']

    # Load arguments from config
    if args.config:
        with open(args.config) as f:
            init_config = json.load(f)

    # Check if required args are provided
    for arg in required_args:
        arg_present = arg in init_config.keys() or arg_dict[arg] is not None
        assert arg_present, parser.error(f'{arg} not provided in config json or as cmd arg')

    # Overwrite provided config values with commandline args, or provide all args at once
    for arg, val in arg_dict.items():
        if val is not None and arg != 'config':
            init_config[arg] = val
            if args.config:
                print(f'Overwriting {arg} value that was provided in {args.config}')

    # Cast activation names to (layer, name) format
    for i, name in enumerate(init_config['activation_names']):
        init_config['activation_names'][i] = int(name[-1]), name[0:-1]

    print()
    pprint(init_config)
    print()

    return init_config


if __name__ == '__main__':
    config = load_config()

    cutoff = config['cutoff']
    print_every = config['print_every']
    del config['cutoff']
    del config['print_every']

    extractor = Extractor(**config)
    extractor.extract(cutoff, print_every)
