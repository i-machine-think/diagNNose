import os
from typing import Dict, Optional, Tuple

import torch
from overrides import overrides
from torch import Tensor

from diagnnose.activations.init_states import InitStates
from diagnnose.typedefs.activations import LayeredTensorDict, TensorDict
from diagnnose.typedefs.models import LanguageModel


class ForwardLSTM(LanguageModel):
    """ Defines a default uni-directional n-layer LSTM.

    Allows for extraction of intermediate states and gate activations.

    Parameters
    ----------
    init_lstm_states_path: str, optional
        Path to pickled initial embeddings

    """

    array_type = "torch"
    ih_concat_order = ["h", "i"]
    split_order = ["i", "f", "g", "o"]

    # TODO: add documentation for init params
    def __init__(
        self,
        state_dict: str,
        init_lstm_states_path: Optional[str] = None,
        device: str = "cpu",
        rnn_name: str = "rnn",
        encoder_name: str = "encoder",
        decoder_name: str = "decoder",
    ) -> None:
        super().__init__()
        print("Loading pretrained model...")

        with open(os.path.expanduser(state_dict), "rb") as mf:
            params: Dict[str, Tensor] = torch.load(mf, map_location=device)

        self.device: str = device
        self.weight: LayeredTensorDict = {}
        self.bias: LayeredTensorDict = {}

        # LSTM weights
        layer = 0
        assert (
            self.rnn_names(0, rnn_name)["weight_hh"] in params
        ), "rnn weight name not found, check if setup is correct"

        while self.rnn_names(layer, rnn_name)["weight_hh"] in params:
            rnn_names = self.rnn_names(layer, rnn_name)

            w_h = params[rnn_names["weight_hh"]]
            w_i = params[rnn_names["weight_ih"]]

            # (2*hidden_size, 4*hidden_size)
            self.weight[layer] = torch.cat((w_h, w_i), dim=1)

            if rnn_names["bias_hh"] in params:
                # (4*hidden_size,)
                self.bias[layer] = (
                    params[rnn_names["bias_hh"]] + params[rnn_names["bias_ih"]]
                )

            self.sizes[layer] = {"x": w_i.size(1), "h": w_h.size(1), "c": w_h.size(1)}
            layer += 1

        # Encoder and decoder weights
        # self.vocab = W2I(create_vocab_from_path(vocab_path))
        self.encoder = params[f"{encoder_name}.weight"]
        self.decoder_w = params[f"{decoder_name}.weight"]
        if f"{decoder_name}.bias" in params:
            self.decoder_b = params[f"{decoder_name}.bias"]
        else:
            self.decoder_b = None

        self.init_lstm_states: InitStates = InitStates(self, init_lstm_states_path)

        print("Model initialisation finished.")

    def forward_step(
        # (4*hidden_size,)
        ih_concat = torch.cat((prev_hx, inp), dim=1)
        self, layer: int, emb: Tensor, prev_hx: Tensor, prev_cx: Tensor
    ) -> TensorDict:
        proj = ih_concat @ self.weight[layer].t()
        if layer in self.bias:
            proj += self.bias[layer]
        split_proj = dict(
            zip(self.split_order, torch.split(proj, self.sizes[layer]["c"], dim=1))
        )

        f_g: Tensor = torch.sigmoid(split_proj["f"])
        i_g: Tensor = torch.sigmoid(split_proj["i"])
        o_g: Tensor = torch.sigmoid(split_proj["o"])
        c_tilde_g: Tensor = torch.tanh(split_proj["g"])

        cx: Tensor = f_g * prev_cx + i_g * c_tilde_g
        hx: Tensor = o_g * torch.tanh(cx)

        return {
            (layer, "emb"): emb,
            (layer, "hx"): hx,
            (layer, "cx"): cx,
            (layer, "f_g"): f_g,
            (layer, "i_g"): i_g,
            (layer, "o_g"): o_g,
            (layer, "c_tilde_g"): c_tilde_g,
        }

    @overrides
    def forward(
        self,
        input_: Tensor,
        prev_activations: Optional[TensorDict] = None,
        compute_out: bool = True,
    ) -> Tuple[Optional[Tensor], TensorDict]:

        # Look up the embeddings of the input words
        embs = self.encoder[input_]

        if prev_activations is None:
            bsz = embs.size(0)
            prev_activations = self.init_hidden(bsz)

        # Iteratively compute and store intermediate rnn activations
        activations: TensorDict = {}
        for l in range(self.num_layers):
            prev_hx = prev_activations[l, "hx"]
            prev_cx = prev_activations[l, "cx"]
            activations.update(self.forward_step(l, embs, prev_hx, prev_cx))
            embs = activations[l, "hx"]

        out: Optional[Tensor] = None
        if compute_out:
            out = self.decoder_w @ input_
            if self.decoder_b is not None:
                out += self.decoder_b

        return out, activations

    @staticmethod
    def rnn_names(layer: int, rnn_name: str) -> Dict[str, str]:
        return {
            "weight_hh": f"{rnn_name}.weight_hh_l{layer}",
            "weight_ih": f"{rnn_name}.weight_ih_l{layer}",
            "bias_hh": f"{rnn_name}.bias_hh_l{layer}",
            "bias_ih": f"{rnn_name}.bias_ih_l{layer}",
        }

    def init_hidden(self, bsz: int) -> TensorDict:
        return self.init_lstm_states.create(bsz)

    def final_hidden(self, hidden: TensorDict) -> torch.Tensor:
        return hidden[self.num_layers - 1, "hx"].squeeze()
