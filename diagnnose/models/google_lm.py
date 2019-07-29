import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from overrides import overrides
from torch import Tensor

from diagnnose.typedefs.activations import ActivationTensors, LayeredTensors
from diagnnose.typedefs.lm import LanguageModel
from diagnnose.vocab import C2I, create_vocab_from_corpus, create_vocab_from_path


class GoogleLM(LanguageModel):
    """ Reimplementation of the LM of Jozefowicz et al. (2016).

    Paper: https://arxiv.org/abs/1602.02410
    Lib: https://github.com/tensorflow/models/tree/master/research/lm_1b

    This implementation allows for only a subset of the SoftMax to be
    loaded in, to alleviate RAM usage.
    """

    forget_offset = 1
    ih_concat_order = ["i", "h"]
    sizes = {l: {"x": 1024, "h": 1024, "c": 8192} for l in range(2)}
    split_order = ["i", "g", "f", "o"]

    def __init__(
        self,
        pbtxt_path: str,
        ckpt_dir: str,
        full_vocab_path: str,
        corpus_vocab_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        print("Loading pretrained model...")

        if corpus_vocab_path is None:
            vocab: C2I = create_vocab_from_path(full_vocab_path, create_char_vocab=True)
        else:
            vocab = create_vocab_from_corpus(corpus_vocab_path, create_char_vocab=True)

        self.encoder = CharCNN(pbtxt_path, ckpt_dir, vocab)
        self.lstm = LSTM(
            ckpt_dir, self.num_layers, self.split_order, self.forget_offset
        )
        self.decoder = SoftMax(vocab, full_vocab_path, ckpt_dir, self.sizes[1]["h"])

        print("Model initialisation finished.")

    @property
    def vocab(self) -> C2I:
        return self.encoder.vocab

    @property
    def weight(self) -> LayeredTensors:
        return self.lstm.weight

    @property
    def bias(self) -> LayeredTensors:
        return self.lstm.bias

    @property
    def peepholes(self) -> ActivationTensors:
        return self.lstm.peepholes

    @property
    def decoder_w(self) -> Tensor:
        return self.decoder.decoder_w

    @property
    def decoder_b(self) -> Tensor:
        return self.decoder.decoder_b

    @overrides
    def forward(
        self, token: str, prev_activations: ActivationTensors, compute_out: bool = True
    ) -> Tuple[Optional[Tensor], ActivationTensors]:
        # Create the embeddings of the input words
        embs = self.encoder.encode(token)

        logits, activations = self.lstm(embs, prev_activations)

        if compute_out:
            out = self.decoder_w @ logits + self.decoder_b
        else:
            out = None

        return out, activations


class CharCNN:
    def __init__(self, pbtxt_path: str, ckpt_dir: str, vocab: C2I) -> None:
        print("Loading CharCNN...")

        self.cnn_sess, self.cnn_t = self._load_char_cnn(pbtxt_path, ckpt_dir)
        self.cnn_embs: Dict[str, Tensor] = {}
        self.vocab = vocab

    @staticmethod
    def _load_char_cnn(pbtxt_path: str, ckpt_dir: str) -> Any:
        import tensorflow as tf
        from google.protobuf import text_format

        ckpt_file = os.path.join(ckpt_dir, "ckpt-char-embedding")

        with tf.Graph().as_default():
            sys.stderr.write("Recovering graph.\n")
            with tf.gfile.FastGFile(pbtxt_path, "r") as f:
                s = f.read()
                gd = tf.GraphDef()
                text_format.Merge(s, gd)

            t = dict()
            [t["char_inputs_in"], t["all_embs"]] = tf.import_graph_def(
                gd, {}, ["char_inputs_in:0", "all_embs_out:0"], name=""
            )

            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            sess.run(f"save/Assign", {"save/Const:0": ckpt_file})
            # The following was recovered from the graph structure, the first 62 assign modules
            # relate to the parameters of the char CNN.
            for i in range(1, 62):
                sess.run(f"save/Assign_{i}", {"save/Const:0": ckpt_file})

        return sess, t

    def encode(self, token: str) -> Tensor:
        if token in self.cnn_embs:
            return self.cnn_embs[token]

        input_dict = {
            self.cnn_t["char_inputs_in"]: self.vocab.word_to_char_ids(token).reshape(
                [-1, 1, self.vocab.max_word_length]
            )
        }
        emb = torch.from_numpy(
            self.cnn_sess.run(self.cnn_t["all_embs"], input_dict)[0]
        ).to(torch.float32)

        self.cnn_embs[token] = emb

        return emb


class LSTM(nn.Module):
    def __init__(
        self, ckpt_dir: str, num_layers: int, split_order: List[str], forget_offset: int
    ) -> None:
        super().__init__()

        print("Loading LSTM...")

        self.num_layers = num_layers
        self.split_order = split_order
        self.forget_offset = forget_offset

        # Projects hidden+input (2*1024) onto cell state dimension (8192)
        self.weight: LayeredTensors = {}
        self.bias: LayeredTensors = {}

        # Projects cell state dimension (8192) back to hidden dimension (1024)
        self.weight_P: LayeredTensors = {}
        # The 3 peepholes are weighted by a diagonal matrix
        self.peepholes: ActivationTensors = {}

        self._load_lstm(ckpt_dir)

    def _load_lstm(self, ckpt_dir: str) -> None:
        from tensorflow.python.pywrap_tensorflow import NewCheckpointReader

        lstm_reader = NewCheckpointReader(os.path.join(ckpt_dir, "ckpt-lstm"))

        for l in range(self.num_layers):
            # Model weights are divided into 8 chunks
            # Shape: (32768, 2048)
            self.weight[l] = torch.stack(
                [
                    torch.from_numpy(
                        lstm_reader.get_tensor(f"lstm/lstm_{l}/W_{i}") for i in range(8)
                    )
                ]
            ).t()

            # Shape: (32768,)
            self.bias[l] = torch.from_numpy(lstm_reader.get_tensor(f"lstm/lstm_{l}/B"))

            # Shape: (8192, 1024)
            self.weight_P[l] = torch.stack(
                [
                    torch.from_numpy(
                        lstm_reader.get_tensor(f"lstm/lstm_{l}/W_P_{i}")
                        for i in range(8)
                    )
                ]
            )

            for p in ["f", "i", "o"]:
                # Shape: (8192, 8192)
                self.peepholes[l, p] = torch.from_numpy(
                    lstm_reader.get_tensor(f"lstm/lstm_{l}/W_{p.upper()}_diag")
                )

            # Cast to float32 tensors
            self.weight[l] = self.weight[l].to(torch.float32)
            self.weight_P[l] = self.weight_P[l].to(torch.float32)
            self.bias[l] = self.bias[l].to(torch.float32)
            for p in ["f", "i", "o"]:
                self.peepholes[l, p] = self.peepholes[l, p].to(torch.float32)

    def forward_step(
        self, layer: int, emb: Tensor, prev_hx: Tensor, prev_cx: Tensor
    ) -> ActivationTensors:
        proj: Tensor = self.weight[layer] @ torch.cat((emb, prev_hx), dim=1)
        proj += self.bias[layer]

        split_proj: Dict[str, Tensor] = dict(
            zip(self.split_order, torch.split(proj, 4, dim=1))
        )

        f_g = torch.sigmoid(
            split_proj["f"] + prev_cx * self.peepholes[layer, "f"] + self.forget_offset
        )
        i_g = torch.sigmoid(split_proj["i"] + prev_cx * self.peepholes[layer, "i"])
        c_tilde_g = torch.tanh(split_proj["g"])

        cx = f_g * prev_cx + i_g * c_tilde_g
        o_g = torch.sigmoid(split_proj["o"] + cx * self.peepholes[layer, "o"])
        hx = (o_g * torch.tanh(cx)) @ self.weight_P[layer]

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
        self, input_: Tensor, prev_activations: ActivationTensors
    ) -> Tuple[Optional[Tensor], ActivationTensors]:
        # Iteratively compute and store intermediate rnn activations
        activations: ActivationTensors = {}

        for l in range(self.num_layers):
            prev_hx = prev_activations[l, "hx"]
            prev_cx = prev_activations[l, "cx"]
            activations = self.forward_step(l, input_, prev_hx, prev_cx)
            input_ = activations[l, "hx"]

        return input_, activations


class SoftMax:
    def __init__(
        self, vocab: C2I, full_vocab_path: str, ckpt_dir: str, hidden_size_h: int
    ) -> None:
        print("Loading SoftMax...")
        self.decoder_w: Tensor = torch.zeros(
            (len(vocab), hidden_size_h), dtype=torch.float32
        )
        self.decoder_b: Tensor = torch.zeros(len(vocab), dtype=torch.float32)

        self._load_softmax(vocab, full_vocab_path, ckpt_dir)

    def _load_softmax(self, vocab: C2I, full_vocab_path: str, ckpt_dir: str) -> None:
        from tensorflow.python.pywrap_tensorflow import NewCheckpointReader

        with open(full_vocab_path) as f:
            full_vocab: List[str] = f.read().strip().split("\n")

        bias_reader = NewCheckpointReader(os.path.join(ckpt_dir, "ckpt-softmax8"))
        full_bias = bias_reader.get_tensor("softmax/b")

        # SoftMax is chunked into 8 arrays of size 100000x1024
        for i in range(8):
            sm_reader = NewCheckpointReader(os.path.join(ckpt_dir, f"ckpt-softmax{i}"))

            sm_chunk = torch.from_numpy(sm_reader.get_tensor(f"softmax/W_{i}")).to(
                torch.float32
            )
            bias_chunk = full_bias[i : len(full_bias) : 8]
            vocab_chunk = full_vocab[i : len(full_bias) : 8]

            for j, w in enumerate(vocab_chunk):
                sm = sm_chunk[j]
                bias = bias_chunk[j]

                if w in vocab:
                    self.decoder_w[vocab[w]] = sm
                    self.decoder_b[vocab[w]] = bias

                if w == "</S>":
                    self.decoder_w[vocab[vocab.eos_token]] = sm
                    self.decoder_b[vocab[vocab.eos_token]] = bias
                elif w == "<UNK>":
                    self.decoder_w[vocab[vocab.unk_token]] = sm
                    self.decoder_b[vocab[vocab.unk_token]] = bias
