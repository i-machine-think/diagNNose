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
                          help='Location of json file with model setup')
    from_cmd.add_argument('--vocab',
                          help='Location of model vocabulary')
    from_cmd.add_argument('--corpus',
                          help='Location of labeled corpus')
    from_cmd.add_argument('--load_modules',
                          help='Folder containing modules that should be loaded to load the model')
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
    print(arg_dict)
    # load arguments from config
    # TODO: Allow overwriting config json
    if args.config:
        # check if config file is the only argument given
        assert len([arg for arg in arg_dict if arg_dict[arg] is not None]) == 1, \
            parser.error("--config file cannot be used in combination with other command line args")
        with open(args.config) as f:
            config = json.load(f)

    # specify arguments via commandline
    else:
        assert args.model,  parser.error("--model is required")
        assert args.vocab,  parser.error("--vocab is required")
        assert args.corpus, parser.error("--corpus is required")
        assert args.output_dir, parser.error("--output_dir is required")
        config = arg_dict
        del arg_dict['config']

    config['activation_names'] = [(int(x[-1]), x[0:-1]) for x in config['activation_names']]

    pprint(config)

    return config


if __name__ == '__main__':
    config = load_config()

    cutoff = config['cutoff']
    print_every = config['print_every']
    del config['cutoff']
    del config['print_every']

    extractor = Extractor(**config)
    extractor.extract(cutoff, print_every)
