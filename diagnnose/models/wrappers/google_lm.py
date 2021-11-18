import os
import sys
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor

from diagnnose.models.recurrent_lm import RecurrentLM
from diagnnose.tokenizer import create_char_vocab
from diagnnose.tokenizer.c2i import C2I


class GoogleLM(RecurrentLM):
    """Reimplementation of the LM of Jozefowicz et al. (2016).

    Paper: https://arxiv.org/abs/1602.02410
    Lib: https://github.com/tensorflow/models/tree/master/research/lm_1b

    This implementation allows for only a subset of the SoftMax to be
    loaded in, to alleviate RAM usage.

    Parameters
    ----------
    ckpt_dir : str
        Path to folder containing parameter checkpoint files.
    corpus_vocab_path : str, optional
        Path to the corpus for which a vocabulary will be created. This
        allows for only a subset of the model softmax to be loaded in.
    create_decoder : bool
        Toggle to load in the (partial) softmax weights. Can be set to
        false in case no decoding projection needs to be made, as is
        the case during activation extraction, for example.
    """

    sizes = {
        (layer, name): size
        for layer in range(2)
        for name, size in [("emb", 1024), ("hx", 1024), ("cx", 8192)]
    }
    forget_offset = 1
    ih_concat_order = ["i", "h"]
    split_order = ["i", "g", "f", "o"]
    use_char_embs = True
    use_peepholes = True

    def __init__(
        self,
        ckpt_dir: str,
        corpus_vocab_path: Optional[Union[str, List[str]]] = None,
        create_decoder: bool = True,
        device: str = "cpu",
    ) -> None:
        super().__init__(device)
        print("Loading pretrained model...")

        full_vocab_path = os.path.join(ckpt_dir, "vocab-2016-09-10.txt")
        vocab: C2I = create_char_vocab(
            corpus_vocab_path or full_vocab_path, unk_token="<UNK>"
        )

        self.encoder = CharCNN(ckpt_dir, vocab, device)

        self._set_lstm_weights(ckpt_dir, device)

        if create_decoder:
            self.decoder = SoftMax(
                vocab, full_vocab_path, ckpt_dir, self.sizes[1, "hx"], device
            )
            self.decoder_w = self.decoder.decoder_w
            self.decoder_b = self.decoder.decoder_b

        print("Model initialisation finished.")

    def create_inputs_embeds(self, input_ids: Tensor) -> Tensor:
        return self.encoder(input_ids)

    def decode(self, hidden_state: Tensor) -> Tensor:
        return self.decoder_w @ hidden_state + self.decoder_b

    def _set_lstm_weights(self, ckpt_dir: str, device: str) -> None:
        from tensorflow.compat.v1.train import NewCheckpointReader

        print("Loading LSTM...")

        lstm_reader = NewCheckpointReader(os.path.join(ckpt_dir, "ckpt-lstm"))

        for l in range(self.num_layers):
            # Model weights are divided into 8 chunks
            # Shape: (2048, 32768)
            self.weight[l] = torch.cat(
                [
                    torch.from_numpy(lstm_reader.get_tensor(f"lstm/lstm_{l}/W_{i}"))
                    for i in range(8)
                ],
                dim=0,
            )

            # Shape: (32768,)
            self.bias[l] = torch.from_numpy(lstm_reader.get_tensor(f"lstm/lstm_{l}/B"))

            # Shape: (8192, 1024)
            self.weight_P[l] = torch.cat(
                [
                    torch.from_numpy(lstm_reader.get_tensor(f"lstm/lstm_{l}/W_P_{i}"))
                    for i in range(8)
                ],
                dim=0,
            )

            for p in ["f", "i", "o"]:
                # Shape: (8192, 8192)
                self.peepholes[l, p] = torch.from_numpy(
                    lstm_reader.get_tensor(f"lstm/lstm_{l}/W_{p.upper()}_diag")
                )

            # Cast to float32 tensors
            self.weight[l] = self.weight[l].to(device)
            self.weight_P[l] = self.weight_P[l].to(device)
            self.bias[l] = self.bias[l].to(device)
            for p in ["f", "i", "o"]:
                self.peepholes[l, p] = self.peepholes[l, p].to(device)


class CharCNN(nn.Module):
    def __init__(self, ckpt_dir: str, vocab: C2I, device: str) -> None:
        print("Loading CharCNN...")
        super().__init__()

        self.cnn_sess, self.cnn_t = self._load_char_cnn(ckpt_dir)
        self.cnn_embs: Dict[str, Tensor] = {}
        self.vocab = vocab
        self.device = device

    @staticmethod
    def _load_char_cnn(ckpt_dir: str) -> Any:
        import tensorflow as tf
        from google.protobuf import text_format

        pbtxt_file = os.path.join(ckpt_dir, "graph-2016-09-10.pbtxt")
        ckpt_file = os.path.join(ckpt_dir, "ckpt-char-embedding")

        with tf.compat.v1.Graph().as_default():
            sys.stderr.write("Recovering graph.\n")
            with tf.compat.v1.gfile.FastGFile(pbtxt_file, "r") as f:
                s = f.read()
                gd = tf.compat.v1.GraphDef()
                text_format.Merge(s, gd)

            t = dict()
            [t["char_inputs_in"], t["all_embs"]] = tf.import_graph_def(
                gd, {}, ["char_inputs_in:0", "all_embs_out:0"], name=""
            )

            sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(allow_soft_placement=True)
            )
            sess.run(f"save/Assign", {"save/Const:0": ckpt_file})
            # The following was recovered from the graph structure, the first 62 assign modules
            # relate to the parameters of the char CNN.
            for i in range(1, 62):
                sess.run(f"save/Assign_{i}", {"save/Const:0": ckpt_file})

        return sess, t

    def forward(self, input_ids: Tensor) -> Tensor:
        """Fetches the character-CNN embeddings of a batch

        Parameters
        ----------
        input_ids : (batch_size, max_sen_len)

        Returns
        -------
        inputs_embeds : (batch_size, max_sen_len, emb_dim)
        """
        inputs_embeds = torch.zeros((*input_ids.shape, 1024), device=self.device)

        for batch_idx in range(input_ids.shape[0]):
            tokens: List[str] = [
                self.vocab.i2w[token_idx] for token_idx in input_ids[batch_idx]
            ]

            for i, token in enumerate(tokens):
                if token not in self.cnn_embs:
                    char_ids = self.vocab.token_to_char_ids(token)
                    input_dict = {self.cnn_t["char_inputs_in"]: char_ids}
                    emb = torch.from_numpy(
                        self.cnn_sess.run(self.cnn_t["all_embs"], input_dict)
                    ).to(self.device)
                    self.cnn_embs[token] = emb
                else:
                    emb = self.cnn_embs[token]
                inputs_embeds[batch_idx, i] = emb

        return inputs_embeds


class SoftMax:
    def __init__(
        self,
        vocab: C2I,
        full_vocab_path: str,
        ckpt_dir: str,
        hidden_size_h: int,
        device: str,
    ) -> None:
        print("Loading SoftMax...")
        self.decoder_w: Tensor = torch.zeros((len(vocab), hidden_size_h), device=device)
        self.decoder_b: Tensor = torch.zeros(len(vocab), device=device)

        self._load_softmax(vocab, full_vocab_path, ckpt_dir)

    def _load_softmax(self, vocab: C2I, full_vocab_path: str, ckpt_dir: str) -> None:
        from tensorflow.compat.v1.train import NewCheckpointReader

        with open(full_vocab_path, encoding="ISO-8859-1") as f:
            full_vocab: List[str] = [token.strip() for token in f]

        bias_reader = NewCheckpointReader(os.path.join(ckpt_dir, "ckpt-softmax8"))
        full_bias = torch.from_numpy(bias_reader.get_tensor("softmax/b"))

        # SoftMax is chunked into 8 arrays of size 100000x1024
        for i in range(8):
            sm_reader = NewCheckpointReader(os.path.join(ckpt_dir, f"ckpt-softmax{i}"))

            sm_chunk = torch.from_numpy(sm_reader.get_tensor(f"softmax/W_{i}"))
            bias_chunk = full_bias[i : full_bias.shape[0] : 8]
            vocab_chunk = full_vocab[i : full_bias.shape[0] : 8]

            for j, w in enumerate(vocab_chunk):
                sm = sm_chunk[j]
                bias = bias_chunk[j]

                if w in vocab and vocab[w] < self.decoder_w.shape[0]:
                    self.decoder_w[vocab[w]] = sm
                    self.decoder_b[vocab[w]] = bias
