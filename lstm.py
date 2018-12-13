import torch
from torch import nn
from torch.nn import functional as F


class Forward_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, embed_size, output_size, w2i_dict_path, model_path):
        super(Forward_LSTM, self).__init__()

        self.hidden_size = hidden_size

        with open(w2i_dict_path, 'r') as f:
            vocab_lines = f.readlines()


        self.w2i = {}
        for i, line in enumerate(vocab_lines):
            self.w2i[line.strip()] = i

        self.unk_idx = self.w2i['<unk>']

        # Load the pretrained model
        with open(model_path, 'rb') as f:
            model = torch.load(f, map_location='cpu')

        params = {}
        for name, param in model.named_parameters():
            params[name] = param

        NHID = hidden_size  # only for convenience

        # First layer
        # current timestep
        self.w_ii_l0 = params['rnn.weight_ih_l0'][0:NHID]
        self.w_if_l0 = params['rnn.weight_ih_l0'][NHID:2*NHID]
        self.w_ig_l0 = params['rnn.weight_ih_l0'][2*NHID:3*NHID]
        self.w_io_l0 = params['rnn.weight_ih_l0'][3*NHID:4*NHID]

        self.b_ii_l0 = params['rnn.bias_ih_l0'][0:NHID]
        self.b_if_l0 = params['rnn.bias_ih_l0'][NHID:2*NHID]
        self.b_ig_l0 = params['rnn.bias_ih_l0'][2*NHID:3*NHID]
        self.b_io_l0 = params['rnn.bias_ih_l0'][3*NHID:4*NHID]

        # recursion
        self.b_hi_l0 = params['rnn.bias_hh_l0'][0:NHID]
        self.b_hf_l0 = params['rnn.bias_hh_l0'][NHID:2*NHID]
        self.b_hg_l0 = params['rnn.bias_hh_l0'][2*NHID:3*NHID]
        self.b_ho_l0 = params['rnn.bias_hh_l0'][3*NHID:4*NHID]

        self.w_hi_l0 = params['rnn.weight_hh_l0'][0:NHID]
        self.w_hf_l0 = params['rnn.weight_hh_l0'][NHID:2*NHID]
        self.w_hg_l0 = params['rnn.weight_hh_l0'][2*NHID:3*NHID]
        self.w_ho_l0 = params['rnn.weight_hh_l0'][3*NHID:4*NHID]


        # Second layer
        # current timestep
        self.w_ii_l1 = params['rnn.weight_ih_l1'][0:NHID]
        self.w_if_l1 = params['rnn.weight_ih_l1'][NHID:2*NHID]
        self.w_ig_l1 = params['rnn.weight_ih_l1'][2*NHID:3*NHID]
        self.w_io_l1 = params['rnn.weight_ih_l1'][3*NHID:4*NHID]

        self.b_ii_l1 = params['rnn.bias_ih_l1'][0:NHID]
        self.b_if_l1 = params['rnn.bias_ih_l1'][NHID:2*NHID]
        self.b_ig_l1 = params['rnn.bias_ih_l1'][2*NHID:3*NHID]
        self.b_io_l1 = params['rnn.bias_ih_l1'][3*NHID:4*NHID]

        # recursion
        self.w_hi_l1 = params['rnn.weight_hh_l1'][0:NHID]
        self.w_hf_l1 = params['rnn.weight_hh_l1'][NHID:2*NHID]
        self.w_hg_l1 = params['rnn.weight_hh_l1'][2*NHID:3*NHID]
        self.w_ho_l1 = params['rnn.weight_hh_l1'][3*NHID:4*NHID]

        self.b_hi_l1 = params['rnn.bias_hh_l1'][0:NHID]
        self.b_hf_l1 = params['rnn.bias_hh_l1'][NHID:2*NHID]
        self.b_hg_l1 = params['rnn.bias_hh_l1'][2*NHID:3*NHID]
        self.b_ho_l1 = params['rnn.bias_hh_l1'][3*NHID:4*NHID]

        # Encoder and decoder
        self.encoder = params['encoder.weight']
        self.w_decoder = params['decoder.weight']
        self.b_decoder = params['decoder.bias']


    def forward(self, inp, h0_l0, c0_l0, h0_l1, c0_l1):

        # Look up the embeddings of the input words
        if inp in self.w2i:
            inp = self.encoder[self.w2i[inp]]
        else:
            inp = self.encoder[self.unk_idx]

        # LAYER 0
        # forget gate
        f_g_l0 = F.sigmoid((self.w_if_l0 @ inp + self.b_if_l0) + (self.w_hf_l0 @ h0_l0 + self.b_hf_l0))
        # input gate
        i_g_l0 = F.sigmoid((self.w_ii_l0 @ inp + self.b_ii_l0) + (self.w_hi_l0 @ h0_l0 + self.b_hi_l0))
        # output gate
        o_g_l0 = F.sigmoid((self.w_io_l0 @ inp + self.b_io_l0) + (self.w_ho_l0 @ h0_l0 + self.b_ho_l0))
        # intermediate cell state
        c_tilde_l0 = F.tanh((self.w_ig_l0 @ inp + self.b_ig_l0) + (self.w_hg_l0 @ h0_l0 + self.b_hg_l0))
        # current cell state
        cx_l0 = f_g_l0 * c0_l0 + i_g_l0 * c_tilde_l0
        # hidden state
        hx_l0 = o_g_l0 * F.tanh(cx_l0)


        # LAYER 1
        # forget gate
        f_g_l1 = F.sigmoid((self.w_if_l1 @ hx_l0 + self.b_if_l1) + (self.w_hf_l1 @ h0_l1 + self.b_hf_l1))
        # input gate
        i_g_l1 = F.sigmoid((self.w_ii_l1 @ hx_l0 + self.b_ii_l1) + (self.w_hi_l1 @ h0_l1 + self.b_hi_l1))
        # output gate
        o_g_l1 = F.sigmoid((self.w_io_l1 @ hx_l0  + self.b_io_l1) + (self.w_ho_l1 @ h0_l1 + self.b_ho_l1))
        # intermediate cell state
        c_tilde_l1 = F.tanh((self.w_ig_l1 @ hx_l0 + self.b_ig_l1) + (self.w_hg_l1 @ h0_l1 + self.b_hg_l1))
        # current cell state
        cx_l1 = f_g_l1 * c0_l1 + i_g_l1 * c_tilde_l1
        # hidden state
        hx_l1 = o_g_l1 * F.tanh(cx_l1)

        out = self.w_decoder @ hx_l1 + self.b_decoder

        return out, [hx_l0, cx_l0, f_g_l0, i_g_l0, o_g_l0, c_tilde_l0], [hx_l1, cx_l1, f_g_l1, i_g_l1, o_g_l1, c_tilde_l1]