import torch
import torch.nn.functional as F

def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None): 
    hx, cx = hidden
    gates = F.linear(input, w_ih, b_ih) + F.linear(hx, w_hh, b_hh)

    ingate, forgetgate, cy_tilde, outgate = gates.chunk(4, 1) #dim modified from 1 to 2

    ingate = F.sigmoid(ingate)
    forgetgate = F.sigmoid(forgetgate)
    cy_tilde = F.tanh(cy_tilde)
    outgate = F.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cy_tilde)
    hy = outgate * F.tanh(cy)

    return (hy, cy), {'in': ingate, 'forget': forgetgate, 'out': outgate, 'c_tilde': cy_tilde}

def forward(self, input, hidden):

    self.store = {}

    for i in xrange(self.num_layers):
        self.store[i] = {}

    for key in ['h', 'c', 'in', 'out', 'forget', 'c_tilde']:
        for layer in self.store:
            self.store[layer][key] = []

    output = []
    steps = range(input.size(0))
    for i in steps:
        next_input, hidden = forward_step(self, input[i], hidden)
        # output.append(hidden[0] if isinstance(hidden, tuple) else hidden)
        output.append(next_input)

    output = torch.cat(output, 0)
    # .view(input.size(0), *output[0].size())

    for layer in self.store:
        for key in self.store[layer]:
            self.store[layer][key] = torch.cat(self.store[layer][key])

    return output, hidden


def forward_step(self, input, hidden):
    num_layers = self.num_layers
    weight = self.all_weights
    dropout = self.dropout
    # saves the gate values into the rnn object

    next_hidden = []

    hidden = list(zip(*hidden))

    for l in range(num_layers):

        # we assume there is just one token in the input
        hy, gates = LSTMCell(input[0], hidden[l], *weight[l])

        # store h and c
        self.store[l]['h'].append(hy[0].unsqueeze(0))
        self.store[l]['c'].append(hy[1].unsqueeze(0))

        # store gates
        for gate in ['in', 'out', 'forget', 'c_tilde']:
            self.store[l][gate].append(gates[gate].unsqueeze(0))

        next_hidden.append(hy)

        input = hy[0]

        if dropout != 0 and l < num_layers - 1:
            input = F.dropout(input, p=dropout, training=False, inplace=False)

    next_h, next_c = zip(*next_hidden)
    next_hidden = (
        torch.cat(next_h, 0).view(num_layers, *next_h[0].size()),
        torch.cat(next_c, 0).view(num_layers, *next_c[0].size())
    )


    # we restore the right dimensionality
    input = input.unsqueeze(0)

    return input, next_hidden

