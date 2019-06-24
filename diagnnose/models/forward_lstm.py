import os
from typing import Any, Optional, Tuple

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
                 state_dict: str,
                 vocab_path: str,
                 device: str = 'cpu',
                 rnn_name: str = 'rnn',
                 encoder_name: str = 'encoder',
                 decoder_name: str = 'decoder') -> None:

        super().__init__()

        print('Loading pretrained model...')
        self.vocab = W2I(create_vocab_from_path(vocab_path))

        # Load the pretrained model
        with open(os.path.expanduser(state_dict), 'rb') as mf:
            state_dict = torch.load(mf, map_location=device)

        self.weight: ParameterDict = {}
        self.bias: ParameterDict = {}

        self.num_layers = 0
        while f'{rnn_name}.weight_hh_l{self.num_layers}' in state_dict:
            self.num_layers += 1

        # LSTM weights
        for l in range(self.num_layers):
            # (2*hidden_size, 4*hidden_size)
            self.weight[l] = torch.cat((state_dict[f'{rnn_name}.weight_hh_l{l}'],
                                        state_dict[f'{rnn_name}.weight_ih_l{l}']), dim=1)

            if f'{rnn_name}.bias_ih_l{l}' in state_dict:
                # (4*hidden_size,)
                self.bias[l] = state_dict[f'{rnn_name}.bias_ih_l{l}'] + \
                               state_dict[f'{rnn_name}.bias_hh_l{l}']

        self.hidden_size_c = state_dict[f'{rnn_name}.weight_hh_l0'].size(1)
        self.hidden_size_h = state_dict[f'{rnn_name}.weight_hh_l0'].size(1)
        self.split_order = ['i', 'f', 'g', 'o']
        self.array_type = 'torch'
        self.ih_concat_order = ['h', 'i']

        # Encoder and decoder weights
        self.encoder = state_dict[f'{encoder_name}.weight']
        self.decoder_w = state_dict[f'{decoder_name}.weight']
        if f'{decoder_name}.bias' in state_dict:
            self.decoder_b = state_dict[f'{decoder_name}.bias']
        else:
            self.decoder_b = None

        print('Model initialisation finished.')

    def forward_step(self,
                     l: int,
                     inp: Tensor,
                     prev_hx: Tensor,
                     prev_cx: Tensor) -> NamedArrayDict:
        # (4*hidden_size,)
        proj = self.weight[l] @ torch.cat((prev_hx, inp))
        if l in self.bias:
            proj += self.bias[l]
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
            out = self.decoder_w @ input_
            if self.decoder_b is not None:
                out += self.decoder_b
        else:
            out = None

        return out, activations
