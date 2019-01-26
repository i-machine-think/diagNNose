from argparse import ArgumentParser

import lm_diagnoser
from lm_diagnoser.classifiers.diagnostic_classifier import DiagnosticClassifier
from lm_diagnoser.extractors.base_extractor import Extractor


def init_argparser() -> ArgumentParser:
    parser = ArgumentParser()

    # create group to load config from a file
    from_config = parser.add_argument_group('From file', 'Provide full experiment setup via a configuration file')
    from_config.add_argument('--config',
                        help='Location of json file containing extraction config.')

    # create group to provide info via commandline arguments
    from_commandline = parser.add_argument_group('From commandline', 'Specify experiment setup via commandline arguments')
    from_commandline.add_argument('--model', help='Location of json file with model setup')
    from_commandline.add_argument('--vocab', help='Location of model vocabulary')
    from_commandline.add_argument('--corpus', help='Location of labeled corpus')
    from_commandline.add_argument('--load_modules', help='Folder containing modules that should be loaded to load the model')
    from_commandline.add_argument('--activation_names', help='Activations to be extracted', nargs='*')
    from_commandline.add_argument('--init_embs', help='Location of initial state of the model')
    from_commandline.add_argument('--print_every', type=int, help='Print extraction progress every n steps', default=20)
    from_commandline.add_argument('--cutoff', type=int, help='Stop extraction after n sentences. Defaults to -1 to extract entire corpus', default=-1)
    from_commandline.add_argument('--output_dir', help='Path to which extracted embeddings will be written.')

    return parser


# TODO: allow extractor arguments both directly or from json config
# TODO: Make extraction/classification optional from argparser
if __name__ == '__main__':
    parser = init_argparser()
    args = parser.parse_args()

    arg_dict = vars(args)

    # load arguments from config
    if args.config:

        # check if config file is the only argument given
        assert len([arg for arg in arg_dict if arg_dict[arg] != None]) == 3, parser.error("--config file cannot be used in combination with other command line arguments")
        import json
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


    # TODO: fix for config
    cutoff = config['cutoff']
    del config['cutoff']

    # TODO: HOTFIX
    config['activation_names'] = [(int(x[-1]), x[0:-1]) for x in config['activation_names'] if x[0]]

    print(config)       #TODO disappears abit, perhaps add some whitespace around?

    extractor = Extractor(**config)
    extractor.extract(cutoff)
