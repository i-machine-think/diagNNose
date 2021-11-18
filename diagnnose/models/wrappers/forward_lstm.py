import os
from typing import Dict

import torch
from torch import Tensor

from diagnnose.models.recurrent_lm import RecurrentLM


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
        super().__init__(device)
        print("Loading pretrained model...")

        with open(os.path.expanduser(state_dict), "rb") as mf:
            params: Dict[str, Tensor] = torch.load(mf, map_location=device)

        params = params["state_dict"] if "state_dict" in params else params

        self._set_lstm_weights(params, rnn_name)

        # Encoder and decoder weights
        self.word_embeddings: Tensor = params[f"{encoder_name}.weight"]
        self.decoder_w: Tensor = params[f"{decoder_name}.weight"]
        self.decoder_b: Tensor = params[f"{decoder_name}.bias"]

        self.sizes[self.top_layer, "out"] = self.decoder_b.shape[0]

        print("Model initialisation finished.")

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
