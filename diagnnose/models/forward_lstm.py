import os
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

    array_type = 'torch'
    ih_concat_order = ['h', 'i']
    split_order = ['i', 'f', 'g', 'o']

    def __init__(self,
                 state_dict: str,
                 device: str = 'cpu',
                 rnn_name: str = 'rnn',
                 encoder_name: str = 'encoder',
                 decoder_name: str = 'decoder') -> None:
        super().__init__()
        print('Loading pretrained model...')

        with open(os.path.expanduser(state_dict), 'rb') as mf:
            params: NamedArrayDict = torch.load(mf, map_location=device)

        self.device: str = device
        self.weight: ParameterDict = {}
        self.bias: ParameterDict = {}

        # LSTM weights
        layer = 0
        while f'{rnn_name}.weight_hh_l{layer}' in params:
            w_h = params[f'{rnn_name}.weight_hh_l{layer}']
            w_i = params[f'{rnn_name}.weight_ih_l{layer}']

            # (2*hidden_size, 4*hidden_size)
            self.weight[layer] = torch.cat((w_h, w_i), dim=1)

            if f'{rnn_name}.bias_ih_l{layer}' in params:
                # (4*hidden_size,)
                self.bias[layer] = params[f'{rnn_name}.bias_ih_l{layer}'] + \
                                   params[f'{rnn_name}.bias_hh_l{layer}']

            self.sizes[layer] = {
                'x': w_i.size(1), 'h': w_h.size(1), 'c': w_h.size(1)
            }
            layer += 1

        # Encoder and decoder weights
        # self.vocab = W2I(create_vocab_from_path(vocab_path))
        self.encoder = params[f'{encoder_name}.weight']
        self.decoder_w = params[f'{decoder_name}.weight']
        if f'{decoder_name}.bias' in params:
            self.decoder_b = params[f'{decoder_name}.bias']
        else:
            self.decoder_b = None

        print('Model initialisation finished.')

    def forward_step(self,
                     layer: int,
                     inp: Tensor,
                     prev_hx: Tensor,
                     prev_cx: Tensor) -> NamedArrayDict:
        # (4*hidden_size,)
        ih_concat = torch.cat((prev_hx, inp), dim=1)
        proj = ih_concat @ self.weight[layer].t()
        if layer in self.bias:
            proj += self.bias[layer]
        split_proj = dict(zip(self.split_order, torch.split(proj, self.sizes[layer]['c'], dim=1)))

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
                input_: torch.Tensor,
                prev_activations: FullActivationDict,
                compute_out: bool = True) -> Tuple[Optional[Tensor], FullActivationDict]:

        # Look up the embeddings of the input words
        embs = self.encoder[input_]

        # Iteratively compute and store intermediate rnn activations
        activations: FullActivationDict = {}
        for l in range(self.num_layers):
            prev_hx = prev_activations[l]['hx']
            prev_cx = prev_activations[l]['cx']
            activations[l] = self.forward_step(l, embs, prev_hx, prev_cx)
            embs = activations[l]['hx']

        if compute_out:
            out = self.decoder_w @ input_
            if self.decoder_b is not None:
                out += self.decoder_b
        else:
            out = None

        return out, activations
