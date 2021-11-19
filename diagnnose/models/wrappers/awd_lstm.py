from typing import Any, Dict

from .forward_lstm import ForwardLSTM


class AWDLSTM(ForwardLSTM):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("rnn_name", "rnns")
        super().__init__(*args, **kwargs)

    @staticmethod
    def param_names(
        layer: int, rnn_name: str, no_suffix: bool = False, **kwargs
    ) -> Dict[str, str]:
        # The AWD-LSTM has no separate weight names for a single layer LSTM
        if no_suffix:
            return {
                "weight_hh": "",
                "weight_ih": "",
                "bias_hh": "",
                "bias_ih": "",
            }
        else:
            return {
                "weight_hh": f"{rnn_name}.{layer}.module.weight_hh_l0_raw",
                "weight_ih": f"{rnn_name}.{layer}.module.weight_ih_l0",
                "bias_hh": f"{rnn_name}.{layer}.module.bias_hh_l0",
                "bias_ih": f"{rnn_name}.{layer}.module.bias_ih_l0",
            }
