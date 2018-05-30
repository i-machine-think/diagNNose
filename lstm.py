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

def forward(parent, input, hidden):

    parent.last_gates = []
    parent.last_hidden =[]
    parent.all_outputs = []

    output = []
    steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
    for i in steps:
        hidden = forward_step(parent, input[i], hidden)
        # hack to handle LSTM
        output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

    output = torch.cat(output, 0).view(input.size(0), *output[0].size())

    return hidden, output


def forward_step(parent, input, hidden):
    num_layers = parent.num_layers
    weight = parent.all_weights
    dropout = parent.dropout
    # saves the gate values into the rnn object

    next_hidden = []

    hidden = list(zip(*hidden))

    for l in range(num_layers):
        # we assume there is just one token in the input
        hy, gates = LSTMCell(input[0], hidden[l], *weight[l])
        parent.last_gates.append(gates)
        parent.last_hidden.append(hy)
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

