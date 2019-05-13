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

    # Gulordava ForwardLSTM args
    from_cmd.add_argument('--model_path',
                          help='Path to model parameters')
    from_cmd.add_argument('--vocab_path',
                          help='Path to model vocabulary')
    from_cmd.add_argument('--module_path',
                          help='Path to folder containing model module')
    from_cmd.add_argument('--device',
                          help='(optional) Torch device name on which model will be run.'
                               'Defaults to cpu.')

    # GoogleLM args
    from_cmd.add_argument('--pbtxt_path',
                          help='Path to the .pbtxt file containing the GraphDef model setup.')
    from_cmd.add_argument('--ckpt_dir',
                          help='Path to folder containing parameter checkpoint files.')
    from_cmd.add_argument('--corpus_vocab_path',
                          help='Path to the corpus vocabulary. This allows for only a subset of the'
                               'model softmax to be loaded in.')
    from_cmd.add_argument('--full_vocab_path',
                          help='Path to the full model vocabulary of 800k tokens.')

    # Extracted activation args
    from_cmd.add_argument('--activations_dir',
                          help='Path to folder containing activations to decompose.')
    from_cmd.add_argument('--init_lstm_states_path',
                          help='(optional) Location of initial lstm states of the model. '
                               'If no path is provided zero-initialized states will be used at the'
                               'start of each sequence.')

    # Decomposition args
    from_cmd.add_argument('--decomposer',
                          help='Class name of decomposer constructor. As of now either '
                               'CellDecomposer or ContextualDecomposer')

    return parser


if __name__ == '__main__':
    required_args = {'model_type', 'activations_dir'}
    arg_groups = {
        'decompose': {'decomposer', 'activations_dir', 'num_layers', 'hidden_size',
                      'init_lstm_states_path'},
        'model': {'model_type', 'model_path', 'vocab_path', 'module_path', 'pbtxt_path', 'ckpt_dir',
                  'corpus_vocab_path', 'full_vocab_path', 'device'},
    }
    argparser = init_argparser()

    config_object = ConfigSetup(argparser, required_args, arg_groups)
    config_dict = config_object.config_dict

    model = import_model_from_json(config_dict['model'])

    constructor = DecomposerFactory(model, **config_dict['decompose'])
    decomposer = constructor.create(0, slice(0, 6, 1), classes=[model.vocab['the']])
    cd = decomposer.decompose(1, 2, ['rel-rel', 'rel-b'])
    print(cd['relevant'], cd['irrelevant'])
