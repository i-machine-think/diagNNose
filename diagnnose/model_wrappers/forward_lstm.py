import os
from typing import Dict, Optional, Tuple

import torch
from overrides import overrides
from torch import Tensor

import diagnnose.typedefs.config as config
from diagnnose.models.lm import LanguageModel
from diagnnose.typedefs.activations import ActivationTensors, LayeredTensors


class ForwardLSTM(LanguageModel):
    """ Defines a default uni-directional n-layer LSTM.

    Allows for extraction of intermediate states and gate activations.

    Parameters
    ----------
    state_dict : str
        Path to torch pickle containing the model parameter state dict.
    device : str, optional
        Torch device on which forward passes will be run.
        Defaults to cpu.
    rnn_name : str, optional
        Name of the rnn of the model. Defaults to `rnn`.
    encoder_name : str, optional
        Name of the embedding encoder. Defaults to `encoder`.
    decoder_name : str, optional
        Name of the linear decoder. Defaults to `decoder`.
    """

    ih_concat_order = ["h", "i"]
    split_order = ["i", "f", "g", "o"]

    def __init__(
        self,
        state_dict: str,
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
        self.weight: LayeredTensors = {}
        self.bias: LayeredTensors = {}

        # LSTM weights
        layer = 0
        assert (
            self.rnn_names(0, rnn_name)["weight_hh"] in params
        ), "rnn weight name not found, check if setup is correct"

        while self.rnn_names(layer, rnn_name)["weight_hh"] in params:
            rnn_names = self.rnn_names(layer, rnn_name)

            w_h = params[rnn_names["weight_hh"]]
            w_i = params[rnn_names["weight_ih"]]

            # Shape: (emb_size+nhid_h, 4*nhid_c)
            self.weight[layer] = torch.cat((w_h, w_i), dim=1).t().to(config.DTYPE)

            if rnn_names["bias_hh"] in params:
                # Shape: (4*nhid_c,)
                self.bias[layer] = (
                    params[rnn_names["bias_hh"]] + params[rnn_names["bias_ih"]]
                ).to(config.DTYPE)

            self.sizes[layer] = {"x": w_i.size(1), "h": w_h.size(1), "c": w_h.size(1)}
            layer += 1

        # Encoder and decoder weights
        self.encoder = params[f"{encoder_name}.weight"].to(config.DTYPE)
        self.decoder_w = params[f"{decoder_name}.weight"].to(config.DTYPE)
        self.decoder_b: Optional[Tensor] = None
        if f"{decoder_name}.bias" in params:
            self.decoder_b = params[f"{decoder_name}.bias"].to(config.DTYPE)

        print("Model initialisation finished.")

    def forward_step(
        self, layer: int, emb: Tensor, prev_hx: Tensor, prev_cx: Tensor
    ) -> ActivationTensors:
        """ Performs the forward step of 1 RNN layer.

        Parameters
        ----------
        layer : int
            Current RNN layer.
        emb : Tensor
            Current input embedding. In higher layers this is h^l-1_t.
            Size: bsz x emb_size
        prev_hx : Tensor
            Previous hidden state. Size: bsz x nhid_h
        prev_cx : Tensor
            Previous cell state. Size: bsz x nhid_c

        Returns
        -------
        activations: TensorDict
            Dictionary mapping (layer, name) tuples to tensors.
        """
        # Shape: (bsz, nhid_h+emb_size)
        ih_concat = torch.cat((prev_hx, emb), dim=1)
        # Shape: (bsz, 4*nhid_c)
        proj = ih_concat @ self.weight[layer]
        if layer in self.bias:
            proj += self.bias[layer]

        split_proj: Dict[str, Tensor] = dict(
            zip(self.split_order, torch.split(proj, self.sizes[layer]["c"], dim=1))
        )

        # Shapes: (bsz, nhid_c)
        f_g = torch.sigmoid(split_proj["f"])
        i_g = torch.sigmoid(split_proj["i"])
        o_g = torch.sigmoid(split_proj["o"])
        c_tilde_g = torch.tanh(split_proj["g"])

        cx = f_g * prev_cx + i_g * c_tilde_g
        hx = o_g * torch.tanh(cx)

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
        prev_activations: Optional[ActivationTensors] = None,
        compute_out: bool = True,
    ) -> Tuple[Optional[Tensor], ActivationTensors]:
        """Performs 1 (batched) forward step for a multi-layer RNN.

        Parameters
        ----------
        input_ : Tensor
            Tensor containing a batch of token id's at the current
            sentence position.
        prev_activations : TensorDict, optional
            Dict mapping the activation names of the previous hidden
            and cell states to their corresponding Tensors. Defaults to
            None, indicating the initial states will be used.
        compute_out : bool, optional
            Toggles the computation of the final decoder projection.
            If set to False this projection is not calculated.
            Defaults to True.
        """

        # Look up the embeddings of the input words
        embs = self.encoder[input_]

        if prev_activations is None:
            bsz = embs.size(0)
            prev_activations = self.init_hidden(bsz)

        # Iteratively compute and store intermediate rnn activations
        activations: ActivationTensors = {}
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
