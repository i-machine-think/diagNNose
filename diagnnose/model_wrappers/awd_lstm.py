from typing import Any, Dict

from .forward_lstm import ForwardLSTM


class AWDLSTM(ForwardLSTM):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("rnn_name", "rnns")
        super().__init__(*args, **kwargs)

    @staticmethod
    def rnn_names(layer: int, rnn_name: str) -> Dict[str, str]:
        return {
            "weight_hh": f"{rnn_name}.{layer}.weight_hh_l0",
            "weight_ih": f"{rnn_name}.{layer}.weight_ih_l0",
            "bias_hh": f"{rnn_name}.{layer}.bias_hh_l0",
            "bias_ih": f"{rnn_name}.{layer}.bias_ih_l0",
        }
