import os
import sys
from typing import Tuple

import torch
from overrides import overrides
from torch import Tensor

from diagnnose.typedefs.activations import ActivationLayer, FullActivationDict, ParameterDict

from .language_model import LanguageModel
from .w2i import W2I


class ForwardLSTM(LanguageModel):
    def __init__(self,
                 model_path: str,
                 vocab_path: str,
                 module_path: str,
                 device_name: str = 'cpu') -> None:

        super().__init__(model_path, vocab_path, module_path, device_name)

        sys.path.append(os.path.expanduser(module_path))

        print('Loading pretrained model...')
        with open(os.path.expanduser(vocab_path), 'r') as vf:
            vocab_lines = vf.readlines()

        w2i = {w.strip(): i for i, w in enumerate(vocab_lines)}
        self.w2i = W2I(w2i)

        # Load the pretrained model
        device = torch.device(device_name)
        with open(os.path.expanduser(model_path), 'rb') as mf:
            model = torch.load(mf, map_location=device)

        params = {name: param for name, param in model.named_parameters()}

        self.hidden_size: int = model.rnn.hidden_size
        self.num_layers: int = model.rnn.num_layers
        self.weight: ParameterDict = {}
        self.bias: ParameterDict = {}

        NHID = self.hidden_size

        # LSTM weights
        for l in range(self.num_layers):
            self.weight[l] = {
                'ii': params[f'rnn.weight_ih_l{l}'][0:NHID],
                'if': params[f'rnn.weight_ih_l{l}'][NHID:2*NHID],
                'ig': params[f'rnn.weight_ih_l{l}'][2*NHID:3*NHID],
                'io': params[f'rnn.weight_ih_l{l}'][3*NHID:4*NHID],
                'hi': params[f'rnn.weight_hh_l{l}'][0:NHID],
                'hf': params[f'rnn.weight_hh_l{l}'][NHID:2*NHID],
                'hg': params[f'rnn.weight_hh_l{l}'][2*NHID:3*NHID],
                'ho': params[f'rnn.weight_hh_l{l}'][3*NHID:4*NHID],
            }
            self.bias[l] = {
                'ii': params[f'rnn.bias_ih_l{l}'][0:NHID],
                'if': params[f'rnn.bias_ih_l{l}'][NHID:2*NHID],
                'ig': params[f'rnn.bias_ih_l{l}'][2*NHID:3*NHID],
                'io': params[f'rnn.bias_ih_l{l}'][3*NHID:4*NHID],
                'hi': params[f'rnn.bias_hh_l{l}'][0:NHID],
                'hf': params[f'rnn.bias_hh_l{l}'][NHID:2*NHID],
                'hg': params[f'rnn.bias_hh_l{l}'][2*NHID:3*NHID],
                'ho': params[f'rnn.bias_hh_l{l}'][3*NHID:4*NHID],
            }

        # Encoder and decoder weights
        self.encoder = params['encoder.weight']
        self.w_decoder = params['decoder.weight']
        self.b_decoder = params['decoder.bias']

        print('Model initialisation finished.')

    # TODO: Do LSTM projections in one step?
    def forward_step(self,
                     l: int,
                     inp: Tensor,
                     prev_hx: Tensor,
                     prev_cx: Tensor) -> ActivationLayer:
        # forget gate
        f_g: Tensor = torch.sigmoid(
            (self.weight[l]['if'] @ inp + self.bias[l]['if']) +
            (self.weight[l]['hf'] @ prev_hx + self.bias[l]['hf'])
        )
        # input gate
        i_g: Tensor = torch.sigmoid(
            (self.weight[l]['ii'] @ inp + self.bias[l]['ii']) +
            (self.weight[l]['hi'] @ prev_hx + self.bias[l]['hi'])
        )
        # output gate
        o_g: Tensor = torch.sigmoid(
            (self.weight[l]['io'] @ inp + self.bias[l]['io']) +
            (self.weight[l]['ho'] @ prev_hx + self.bias[l]['ho'])
        )
        # intermediate cell state
        c_tilde_g: Tensor = torch.tanh(
            (self.weight[l]['ig'] @ inp + self.bias[l]['ig']) +
            (self.weight[l]['hg'] @ prev_hx + self.bias[l]['hg'])
        )
        # current cell state
        cx: Tensor = f_g * prev_cx + i_g * c_tilde_g
        # hidden state
        hx: Tensor = o_g * torch.tanh(cx)

        return {
            'emb': inp,
            'hx': hx, 'cx': cx,
            'f_g': f_g, 'i_g': i_g, 'o_g': o_g, 'c_tilde_g': c_tilde_g
        }

    @overrides
    def forward(self,
                token: str,
                prev_activations: FullActivationDict) -> Tuple[Tensor, FullActivationDict]:

        # Look up the embeddings of the input words
        input_ = self.encoder[self.w2i[token]]

        # Iteratively compute and store intermediate rnn activations
        activations: FullActivationDict = {}
        for l in range(self.num_layers):
            prev_hx = prev_activations[l]['hx']
            prev_cx = prev_activations[l]['cx']
            activations[l] = self.forward_step(l, input_, prev_hx, prev_cx)
            input_ = activations[l]['hx']

        out: Tensor = self.w_decoder @ input_ + self.b_decoder

        return out, activations
