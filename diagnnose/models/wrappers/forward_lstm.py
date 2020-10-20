import os
from itertools import product
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from diagnnose.attribute import ShapleyTensor
from diagnnose.models.recurrent_lm import RecurrentLM
from diagnnose.typedefs.activations import (
    ActivationDict,
    ActivationNames,
    LayeredTensors,
)


class ForwardLSTM(RecurrentLM):
    """Defines a default uni-directional n-layer LSTM.

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

        self._set_lstm_weights(params, rnn_name)

        # Encoder and decoder weights
        self.word_embeddings: Tensor = params[f"{encoder_name}.weight"]
        self.decoder_w: Tensor = params[f"{decoder_name}.weight"]
        self.decoder_b: Tensor = params[f"{decoder_name}.bias"]

        self.sizes[self.top_layer, "out"] = self.decoder_b.size(0)

        print("Model initialisation finished.")

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Union[Tensor, ShapleyTensor]] = None,
        input_lengths: Optional[Tensor] = None,
        compute_out: bool = False,
        only_return_top_embs: bool = False,
    ) -> Union[ActivationDict, Tensor]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        if inputs_embeds is None and input_ids is None:
            raise ValueError("inputs_embeds or input_ids must be provided")
        if inputs_embeds is None:
            inputs_embeds = self.create_inputs_embeds(input_ids)
        if len(inputs_embeds.shape) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        iterator, unsorted_indices = self._create_iterator(inputs_embeds, input_lengths)

        all_activations = self._init_activations(inputs_embeds, compute_out)
        cur_activations = self.init_hidden(inputs_embeds.size(0))

        for w_idx, input_ in enumerate(iterator):
            num_input = input_.size(0)
            for a_name in cur_activations:
                cur_activations[a_name] = cur_activations[a_name][:num_input]

            cur_activations = self.forward_step(
                input_, cur_activations, compute_out=compute_out
            )

            for a_name in all_activations:
                all_activations[a_name][:num_input, w_idx] = cur_activations[a_name]

        # Batch had been sorted and needs to be unsorted to retain the original order
        for a_name, activations in all_activations.items():
            all_activations[a_name] = activations[unsorted_indices]

        if only_return_top_embs and compute_out:
            return all_activations[self.top_layer, "out"]
        elif only_return_top_embs:
            return all_activations[self.top_layer, "hx"]

        return all_activations

    def create_inputs_embeds(self, input_ids: Tensor) -> Tensor:
        return self.word_embeddings[input_ids]

    @staticmethod
    def _create_iterator(
        input_ids: Tensor, input_lengths: Optional[Tensor]
    ) -> Tuple[Tuple[Tensor, ...], Tensor]:
        """Creates a PackedSequence that handles batching for the RNN.

        Batch items are sorted based on sentence length, allowing
        <pad> tokens to be skipped efficiently during the forward pass.

        Returns
        -------
        iterator : Tuple[Tensor, ...]
            Tuple of input tensors for each step in the sequence.
        unsorted_indices : Tensor
            Original order of the corpus prior to sorting.
        """
        if input_lengths is None:
            batch_size = input_ids.size(0)
            input_lengths = torch.tensor(batch_size * [input_ids.size(1)])

        packed_batch: PackedSequence = pack_padded_sequence(
            input_ids, lengths=input_lengths, batch_first=True, enforce_sorted=False
        )

        iterator = torch.split(packed_batch.data, list(packed_batch.batch_sizes))

        return iterator, packed_batch.unsorted_indices

    def _init_activations(
        self, inputs_embeds: Tensor, compute_out: bool
    ) -> ActivationDict:
        """Returns a dictionary mapping activation names to tensors.

        If the input is a ShapleyTensor this dict will store the
        ShapleyTensors as well.

        Returns
        -------
        all_activations : ActivationDict
            Dictionary mapping activation names to tensors of shape:
            batch_size x max_sen_len x nhid.
        """
        batch_size, max_sen_len = inputs_embeds.shape[:2]
        all_activations: ActivationDict = {
            a_name: torch.zeros(batch_size, max_sen_len, self.nhid(a_name))
            for a_name in self.activation_names(compute_out)
        }

        if isinstance(inputs_embeds, ShapleyTensor):
            for a_name, activations in all_activations.items():
                all_activations[a_name] = ShapleyTensor(activations)

        return all_activations

    def forward_step(
        self,
        token_embeds: Tensor,
        prev_activations: ActivationDict,
        compute_out: bool = False,
    ) -> ActivationDict:
        """Performs a forward pass of one step across all layers.

        Parameters
        ----------
        token_embeds : Tensor
            Tensor of word embeddings at the current sentence position.
        prev_activations : ActivationDict
            Dict mapping the activation names of the previous hidden
            and cell states to their corresponding Tensors.
        compute_out : bool, optional
            Toggles the computation of the final decoder projection.
            If set to False this projection is not calculated.
            Defaults to True.

        Returns
        -------
        all_activations : ActivationDict
            Dictionary mapping activation names to tensors of shape:
            batch_size x max_sen_len x nhid.
        """
        cur_activations: ActivationDict = {}
        input_ = token_embeds

        for layer in range(self.num_layers):
            prev_hx = prev_activations[layer, "hx"]
            prev_cx = prev_activations[layer, "cx"]

            layer_activations = self.forward_cell(layer, input_, prev_hx, prev_cx)
            cur_activations.update(layer_activations)

            input_ = cur_activations[layer, "hx"]

        if compute_out:
            out = input_ @ self.decoder_w.t()
            out += self.decoder_b
            cur_activations[self.top_layer, "out"] = out

        return cur_activations

    def forward_cell(
        self, layer: int, input_: Tensor, prev_hx: Tensor, prev_cx: Tensor
    ) -> ActivationDict:
        """Performs the forward step of 1 LSTM cell.

        Parameters
        ----------
        layer : int
            Current RNN layer.
        input_ : Tensor
            Current input embedding. In higher layers this is h^l-1_t.
            Size: batch_size x nhid
        prev_hx : Tensor
            Previous hidden state. Size: batch_size x nhid
        prev_cx : Tensor
            Previous cell state. Size: batch_size x nhid

        Returns
        -------
        all_activations : ActivationDict
            Dictionary mapping activation names to tensors of shape:
            batch_size x max_sen_len x nhid.
        """
        # Shape: (bsz, nhid_h+emb_size)
        if self.ih_concat_order == ["h", "i"]:
            ih_concat = torch.cat((prev_hx, input_), dim=1)
        else:
            ih_concat = torch.cat((input_, prev_hx), dim=1)

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

        activation_dict = {
            (layer, "hx"): hx,
            (layer, "cx"): cx,
            (layer, "f_g"): f_g,
            (layer, "i_g"): i_g,
            (layer, "o_g"): o_g,
            (layer, "c_tilde_g"): c_tilde_g,
        }

        if layer == 0:
            activation_dict[0, "emb"] = input_

        return activation_dict

    def _set_lstm_weights(self, params: Dict[str, Tensor], rnn_name: str) -> None:
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
            self.weight[layer] = torch.cat((w_h, w_i), dim=1).t()

            if param_names["bias_hh"] in params:
                # Shape: (4*nhid_c,)
                self.bias[layer] = (
                    params[param_names["bias_hh"]] + params[param_names["bias_ih"]]
                )

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

    @staticmethod
    def param_names(
        layer: int, rnn_name: str, no_suffix: bool = False
    ) -> Dict[str, str]:
        """Creates a dictionary of parameter names in a state_dict.

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
        """Returns a list of all the model's activation names.

        Parameters
        ----------
        compute_out : bool, optional
            Toggles the computation of the final decoder projection.
            If set to False this projection is not calculated.
            Defaults to True.

        Returns
        -------
        activation_names : ActivationNames
            List of (layer, name) tuples.
        """
        lstm_names = ["hx", "cx", "f_g", "i_g", "o_g", "c_tilde_g"]

        activation_names = list(product(range(self.num_layers), lstm_names))
        activation_names.append((0, "emb"))

        if compute_out:
            activation_names.append((self.top_layer, "out"))

        return activation_names
