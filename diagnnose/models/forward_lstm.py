import os
import sys
from typing import Optional, Tuple

import torch
from overrides import overrides
from torch import Tensor

from diagnnose.typedefs.activations import NamedArrayDict, FullActivationDict, ParameterDict

from .language_model import LanguageModel
from diagnnose.utils.vocab import create_vocab_from_path, W2I


class ForwardLSTM(LanguageModel):
    """ Defines a default uni-directional n-layer LSTM.

    Allows for extraction of intermediate states and gate activations.
    """
    def __init__(self,
                 model_path: str,
                 vocab_path: str,
                 module_path: str,
                 device_name: str = 'cpu') -> None:

        super().__init__()

        sys.path.append(os.path.expanduser(module_path))

        print('Loading pretrained model...')
        self.vocab = W2I(create_vocab_from_path(vocab_path))

        # Load the pretrained model
        device = torch.device(device_name)
        with open(os.path.expanduser(model_path), 'rb') as mf:
            model = torch.load(mf, map_location=device)

        params = {name: param for name, param in model.named_parameters()}

        self.num_layers = model.rnn.num_layers
        self.hidden_size_c = model.rnn.hidden_size
        self.hidden_size_h = model.rnn.hidden_size
        self.split_order = ['i', 'f', 'g', 'o']
        self.array_type = 'torch'

        self.weight: ParameterDict = {}
        self.bias: ParameterDict = {}

        # LSTM weights
        for l in range(self.num_layers):
            # (2*hidden_size, 4*hidden_size)
            self.weight[l] = torch.cat((params[f'rnn.weight_hh_l{l}'],
                                        params[f'rnn.weight_ih_l{l}']), dim=1)

            # (4*hidden_size,)
            self.bias[l] = params[f'rnn.bias_ih_l{l}'] + params[f'rnn.bias_hh_l{l}']

        # Encoder and decoder weights
        self.encoder = params['encoder.weight']
        self.decoder_w = params['decoder.weight']
        self.decoder_b = params['decoder.bias']

        print('Model initialisation finished.')

    def forward_step(self,
                     l: int,
                     inp: Tensor,
                     prev_hx: Tensor,
                     prev_cx: Tensor) -> NamedArrayDict:
        # (4*hidden_size,)
        proj = self.weight[l] @ torch.cat((prev_hx, inp)) + self.bias[l]
        split_proj = dict(zip(self.split_order, torch.split(proj, self.hidden_size_c)))

        f_g: Tensor = torch.sigmoid(split_proj['f'])
        i_g: Tensor = torch.sigmoid(split_proj['i'])
        o_g: Tensor = torch.sigmoid(split_proj['o'])
        c_tilde_g: Tensor = torch.tanh(split_proj['g'])

        cx: Tensor = f_g * prev_cx + i_g * c_tilde_g
        hx: Tensor = o_g * torch.tanh(cx)

        return {
            'emb': inp,
            'hx': hx, 'cx': cx,
            'f_g': f_g, 'i_g': i_g, 'o_g': o_g, 'c_tilde_g': c_tilde_g
        }

    @overrides
    def forward(self,
                token: str,
                prev_activations: FullActivationDict,
                compute_out: bool = True) -> Tuple[Optional[Tensor], FullActivationDict]:

        # Look up the embeddings of the input words
        input_ = self.encoder[self.vocab[token]]

        # Iteratively compute and store intermediate rnn activations
        activations: FullActivationDict = {}
        for l in range(self.num_layers):
            prev_hx = prev_activations[l]['hx']
            prev_cx = prev_activations[l]['cx']
            activations[l] = self.forward_step(l, input_, prev_hx, prev_cx)
            input_ = activations[l]['hx']

        if compute_out:
            out = self.decoder_w @ input_ + self.decoder_b
        else:
            out = None

        return out, activations
