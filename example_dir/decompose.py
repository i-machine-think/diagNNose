from argparse import ArgumentParser

from rnnalyse.config.setup import ConfigSetup
from rnnalyse.decompositions.decomposer import Decomposer


def init_argparser() -> ArgumentParser:
    parser = ArgumentParser()

    # create group to load config from a file
    from_config = parser.add_argument_group('From config file',
                                            'Provide full experiment setup via config file')
    from_config.add_argument('-c', '--config',
                             help='Path to json file containing decomposition config.')

    # create group to provide info via commandline arguments
    # Required args are not set to be required here as they can come from --config
    from_cmd = parser.add_argument_group('From commandline',
                                         'Specify experiment setup via commandline arguments')
    from_cmd.add_argument('--activations_dir',
                          help='Path to folder containing activations to decompose.')
    from_cmd.add_argument('--decoder_path',
                          help='Path to decoder classifier.')
    from_cmd.add_argument('--num_layers',
                          help='Number of layers in the language model.')
    from_cmd.add_argument('--hidden_size',
                          help='Size of hidden units in the language model.')
    from_cmd.add_argument('--init_lstm_states_path',
                          help='(optional) Location of initial lstm states of the model. '
                               'If no path is provided zero-initialized states will be used at the'
                               'start of each sequence.')

    return parser


if __name__ == '__main__':
    required_args = {'activations_dir', 'decoder_path', 'num_layers', 'hidden_size'}
    arg_groups = {
        'decompose': {'activations_dir', 'decoder_path', 'num_layers', 'hidden_size',
                      'init_lstm_states_path'},
    }
    argparser = init_argparser()

    config_object = ConfigSetup(argparser, required_args, arg_groups)
    config_dict = config_object.config_dict

    decomposer = Decomposer(**config_dict['decompose'])
    decomposer.decompose(0, 1)