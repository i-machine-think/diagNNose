import torch
from torch import nn


class ForwardLSTM(nn.Module):
    def __init__(self, model_path, vocab_path, device=torch.device('cpu')):
        super(ForwardLSTM, self).__init__()

        with open(vocab_path, 'r') as f:
            vocab_lines = f.readlines()

        self.w2i = {w.strip(): i for i, w in enumerate(vocab_lines)}

        self.unk_idx = self.w2i['<unk>']

        # Load the pretrained model
        with open(model_path, 'rb') as f:
            model = torch.load(f, map_location=device)

        params = {name: param for name, param in model.named_parameters()}

        self.hidden_size = model.rnn.hidden_size
        self.num_layers = model.rnn.num_layers
        self.weight, self.bias = {}, {}

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

    def forward_step(self, l, inp, prev_hx, prev_cx):
        # forget gate
        f_g = torch.sigmoid(
            (self.weight[l]['if'] @ inp + self.bias[l]['if']) + (self.weight[l]['hf'] @ prev_hx + self.bias[l]['hf']))
        # input gate
        i_g = torch.sigmoid(
            (self.weight[l]['ii'] @ inp + self.bias[l]['ii']) + (self.weight[l]['hi'] @ prev_hx + self.bias[l]['hi']))
        # output gate
        o_g = torch.sigmoid(
            (self.weight[l]['io'] @ inp + self.bias[l]['io']) + (self.weight[l]['ho'] @ prev_hx + self.bias[l]['ho']))
        # intermediate cell state
        c_tilde_g = torch.tanh(
            (self.weight[l]['ig'] @ inp + self.bias[l]['ig']) + (self.weight[l]['hg'] @ prev_hx + self.bias[l]['hg']))
        # current cell state
        cx = f_g * prev_cx + i_g * c_tilde_g
        # hidden state
        hx = o_g * torch.tanh(cx)

        return {'hx': hx, 'cx': cx, 'f_g': f_g, 'i_g': i_g, 'o_g': o_g, 'c_tilde_g': c_tilde_g}

    def forward(self, inp, prev_activations):
        # Look up the embeddings of the input words
        if inp in self.w2i:
            inp = self.encoder[self.w2i[inp]]
        else:
            inp = self.encoder[self.unk_idx]

        activations = {}
        for l in range(self.num_layers):
            activations[l] = self.forward_step(l, inp, prev_activations[l]['hx'], prev_activations[l]['cx'])
            inp = activations[l]['hx']

        out = self.w_decoder @ inp + self.b_decoder

        return out, activations
