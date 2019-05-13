from argparse import ArgumentParser

from diagnnose.config.setup import ConfigSetup
from diagnnose.decompositions.factory import DecomposerFactory
from diagnnose.models.import_model import import_model_from_json


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
    from_cmd.add_argument('--decoder',
                          help='Path to decoder classifier.')
    from_cmd.add_argument('--decomposer',
                          help='Class name of decomposer constructor. As of now either '
                               'CellDecomposer or ContextualDecomposer')
    from_cmd.add_argument('--model_path',
                          help='Path to model parameters')
    from_cmd.add_argument('--vocab_path',
                          help='Path to model vocabulary')
    from_cmd.add_argument('--module_path',
                          help='Path to folder containing model module')
    from_cmd.add_argument('--init_lstm_states_path',
                          help='(optional) Location of initial lstm states of the model. '
                               'If no path is provided zero-initialized states will be used at the'
                               'start of each sequence.')

    return parser


if __name__ == '__main__':
    required_args = {'model_type', 'model_path', 'vocab_path', 'module_path', 'activations_dir'}
    arg_groups = {
        'decompose': {'decomposer', 'activations_dir', 'num_layers', 'hidden_size',
                      'init_lstm_states_path'},
        'decoder': {'model_type', 'model_path', 'vocab_path', 'module_path'},
    }
    argparser = init_argparser()

    config_object = ConfigSetup(argparser, required_args, arg_groups)
    config_dict = config_object.config_dict

    model = import_model_from_json(config_dict['decoder'])

    constructor = DecomposerFactory(model, **config_dict['decompose'])
    decomposer = constructor.create(64, slice(0, 6, 1), classes=[model.w2i['is'], model.w2i['are']])
    cd = decomposer.decompose(0, 1, ['rel-rel', 'rel-b'])
    print(cd['relevant'], cd['irrelevant'])
