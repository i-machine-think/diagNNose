import os
from itertools import product
from typing import Dict

import torch
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

import diagnnose.typedefs.config as config
from diagnnose.models.recurrent_lm import RecurrentLM
from diagnnose.typedefs.activations import (
    ActivationDict,
    ActivationNames,
    LayeredTensors,
)


class ForwardLSTM(RecurrentLM):
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
        Name of the rnn in the model state_dict. Defaults to `rnn`.
    encoder_name : str, optional
        Name of the embedding encoder in the model state_dict.
        Defaults to `encoder`.
    decoder_name : str, optional
        Name of the linear decoder in the model state_dict.
        Defaults to `decoder`.
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
        # 1 layer RNNs do not have a layer suffix in the state_dict
        no_suffix = self.param_names(0, rnn_name, no_suffix=True)["weight_hh"] in params
        assert (
            self.param_names(0, rnn_name, no_suffix=no_suffix)["weight_hh"] in params
        ), "rnn weight name not found, check if setup is correct"

        while (
            self.param_names(layer, rnn_name, no_suffix=no_suffix)["weight_hh"]
            in params
        ):
            param_names = self.param_names(layer, rnn_name, no_suffix=no_suffix)

            w_h = params[param_names["weight_hh"]]
            w_i = params[param_names["weight_ih"]]

            # Shape: (emb_size+nhid_h, 4*nhid_c)
            self.weight[layer] = torch.cat((w_h, w_i), dim=1).t().to(config.DTYPE)

            if param_names["bias_hh"] in params:
                # Shape: (4*nhid_c,)
                self.bias[layer] = (
                    params[param_names["bias_hh"]] + params[param_names["bias_ih"]]
                ).to(config.DTYPE)

            self.sizes.update(
                {
                    (layer, "emb"): w_i.size(1),
                    (layer, "hx"): w_h.size(1),
                    (layer, "cx"): w_h.size(1),
                }
            )
            layer += 1

            if no_suffix:
                break

        # Encoder and decoder weights
        self.encoder = params[f"{encoder_name}.weight"].to(config.DTYPE)
        self.decoder_w = params[f"{decoder_name}.weight"].to(config.DTYPE)
        self.decoder_b = params[f"{decoder_name}.bias"].to(config.DTYPE)

        self.sizes[self.top_layer, "out"] = self.decoder_b.size(0)

        print("Model initialisation finished.")

    def forward(
        self, batch: Tensor, batch_lengths: Tensor, compute_out: bool = False
    ) -> ActivationDict:
        """ Performs a single forward pass across all LM layers.

        Parameters
        ----------
        batch : Tensor
            Tensor of a batch of (padded) sentences.
            Size: batch_size x max_sen_len
        batch_lengths : Tensor
            Tensor of the sentence lengths of each batch item.
            Size: batch_size
        compute_out : bool, optional
            Toggles the computation of the final decoder projection.
            If set to False this projection is not calculated.
            Defaults to True.

        Returns
        -------
        activations : ActivationDict
            Dictionary mapping an activation name to a (padded) tensor.
            Size: a_name -> batch_size x max_sen_len x nhid
        """
        packed_batch: PackedSequence = pack_padded_sequence(
            batch, lengths=batch_lengths, batch_first=True, enforce_sorted=False
        )

        # a_name -> batch_size x max_sen_len x nhid
        all_activations: ActivationDict = {
            a_name: torch.zeros(*batch.shape, self.nhid(a_name)).to(config.DTYPE)
            for a_name in self.activation_names(compute_out)
        }
        cur_activations = self.init_hidden(batch.size(0))

        iterator = torch.split(packed_batch.data, list(packed_batch.batch_sizes))

        for w_idx, input_ in enumerate(iterator):
            num_input = packed_batch.batch_sizes[w_idx]
            for a_name in cur_activations:
                cur_activations[a_name] = cur_activations[a_name][:num_input]

            cur_activations = self.forward_step(
                input_, cur_activations, compute_out=compute_out
            )

            for a_name in all_activations:
                all_activations[a_name][:num_input, w_idx] = cur_activations[a_name]

        for a_name, activations in all_activations.items():
            all_activations[a_name] = activations[packed_batch.unsorted_indices]

        return all_activations

    def forward_step(
        self,
        input_: Tensor,
        prev_activations: ActivationDict,
        compute_out: bool = False,
    ) -> ActivationDict:
        """Performs a (batched) forward pass for a multi-layer RNN.

        Parameters
        ----------
        input_ : Tensor
            Tensor containing a batch of token id's at the current
            sentence position.
        prev_activations : ActivationDict
            Dict mapping the activation names of the previous hidden
            and cell states to their corresponding Tensors.
        compute_out : bool, optional
            Toggles the computation of the final decoder projection.
            If set to False this projection is not calculated.
            Defaults to True.

        Returns
        -------
        cur_activations : ActivationDict
            Dictionary mapping an activation name to its current
            activation.
        """
        # Look up the embeddings of the input tokens
        input_ = self.encoder[input_]

        # Iteratively compute and store intermediate rnn activations
        cur_activations: ActivationDict = {}
        for layer in range(self.num_layers):
            prev_hx = prev_activations[layer, "hx"]
            prev_cx = prev_activations[layer, "cx"]
            cur_activations.update(self.forward_cell(layer, input_, prev_hx, prev_cx))
            input_ = cur_activations[layer, "hx"]

        if compute_out:
            out = input_ @ self.decoder_w.t()
            out += self.decoder_b
            cur_activations[self.top_layer, "out"] = out

        return cur_activations

    def forward_cell(
        self, layer: int, input_: Tensor, prev_hx: Tensor, prev_cx: Tensor
    ) -> ActivationDict:
        """ Performs the forward step of 1 RNN layer.

        Parameters
        ----------
        layer : int
            Current RNN layer.
        input_ : Tensor
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
        ih_concat = torch.cat((prev_hx, input_), dim=1)
        # Shape: (bsz, 4*nhid_c)
        proj = ih_concat @ self.weight[layer]
        if layer in self.bias:
            proj += self.bias[layer]

        split_proj: Dict[str, Tensor] = dict(
            zip(self.split_order, torch.split(proj, self.sizes[layer, "cx"], dim=1))
        )

        # Shapes: (bsz, nhid_c)
        f_g = torch.sigmoid(split_proj["f"])
        i_g = torch.sigmoid(split_proj["i"])
        o_g = torch.sigmoid(split_proj["o"])
        c_tilde_g = torch.tanh(split_proj["g"])

        cx = f_g * prev_cx + i_g * c_tilde_g
        hx = o_g * torch.tanh(cx)

        return {
            (layer, "emb"): input_,
            (layer, "hx"): hx,
            (layer, "cx"): cx,
            (layer, "f_g"): f_g,
            (layer, "i_g"): i_g,
            (layer, "o_g"): o_g,
            (layer, "c_tilde_g"): c_tilde_g,
        }

    @staticmethod
    def param_names(
        layer: int, rnn_name: str, no_suffix: bool = False
    ) -> Dict[str, str]:
        """ Creates a dictionary of parameter names in a state_dict.

        Parameters
        ----------
        layer : int
            Current layer index.
        rnn_name : str
            Name of the rnn in the model state_dict. Defaults to `rnn`.
        no_suffix : bool, optional
            Toggle to omit the `_l{layer}` suffix from a parameter name.
            1-layer RNNs do not have this suffix. Defaults to False.

        Returns
        -------
        param_names : Dict[str, str]
            Dictionary mapping a general parameter name to the model
            specific parameter name.
        """
        if no_suffix:
            return {
                "weight_hh": f"{rnn_name}.weight_hh",
                "weight_ih": f"{rnn_name}.weight_ih",
                "bias_hh": f"{rnn_name}.bias_hh",
                "bias_ih": f"{rnn_name}.bias_ih",
            }

        return {
            "weight_hh": f"{rnn_name}.weight_hh_l{layer}",
            "weight_ih": f"{rnn_name}.weight_ih_l{layer}",
            "bias_hh": f"{rnn_name}.bias_hh_l{layer}",
            "bias_ih": f"{rnn_name}.bias_ih_l{layer}",
        }

    def activation_names(self, compute_out: bool = False) -> ActivationNames:
        lstm_names = ["emb", "hx", "cx", "f_g", "i_g", "o_g", "c_tilde_g"]

        activation_names = list(product(range(self.num_layers), lstm_names))

        if compute_out:
            activation_names.append((self.top_layer, "out"))

        return activation_names
