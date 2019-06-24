import os
import sys
from typing import Any, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import torch.nn as nn
from google.protobuf import text_format
from overrides import overrides
from scipy.special import expit as sigmoid
from tensorflow.python.pywrap_tensorflow import NewCheckpointReader

from diagnnose.typedefs.activations import (
    FullActivationDict, NamedArrayDict, ParameterDict, PartialArrayDict)
from diagnnose.utils.vocab import C2I, create_vocab_from_corpus, create_vocab_from_path

from .language_model import LanguageModel


class GoogleLM(LanguageModel):
    def __init__(self,
                 pbtxt_path: str,
                 ckpt_dir: str,
                 full_vocab_path: str,
                 corpus_vocab_path: Optional[str] = None) -> None:
        super().__init__()

        self.num_layers = 2
        self.hidden_size_c = 8192
        self.hidden_size_h = 1024
        self.split_order = ['i', 'g', 'f', 'o']
        # TODO: port this model to pytorch, this is a torch lib after all...
        self.array_type = 'numpy'
        self.forget_offset = 1
        self.ih_concat_order = ['i', 'h']

        if corpus_vocab_path is None:
            vocab = C2I(create_vocab_from_path(full_vocab_path))
        else:
            vocab = C2I(create_vocab_from_corpus(corpus_vocab_path))

        self.encoder = CharCNN(pbtxt_path, ckpt_dir, vocab)
        self.lstm = LSTM(ckpt_dir, self.num_layers, self.split_order, self.forget_offset)
        self.sm = SoftMax(vocab, full_vocab_path, ckpt_dir, self.hidden_size_h)

        print('Model initialisation finished.')

    @property
    def vocab(self) -> C2I:
        return self.encoder.vocab

    @property
    def weight(self) -> ParameterDict:
        return self.lstm.weight

    @property
    def bias(self) -> ParameterDict:
        return self.lstm.bias

    @property
    def peepholes(self) -> PartialArrayDict:
        return self.lstm.peepholes

    @property
    def decoder_w(self) -> np.ndarray:
        return self.sm.decoder_w

    @property
    def decoder_b(self) -> np.ndarray:
        return self.sm.decoder_b

    @overrides
    def forward(self,
                token: str,
                prev_activations: FullActivationDict,
                compute_out: bool = True) -> Tuple[Optional[np.ndarray], FullActivationDict]:
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
        print('Loading CharCNN...')

        self.cnn_sess, self.cnn_t = self._load_char_cnn(pbtxt_path, ckpt_dir)
        self.cnn_embs: NamedArrayDict = {}
        self.vocab = vocab

    @staticmethod
    def _load_char_cnn(pbtxt_path: str, ckpt_dir: str) -> Any:
        ckpt_file = os.path.join(ckpt_dir, 'ckpt-char-embedding')

        with tf.Graph().as_default():
            sys.stderr.write('Recovering graph.\n')
            with tf.gfile.FastGFile(pbtxt_path, 'r') as f:
                s = f.read()
                gd = tf.GraphDef()
                text_format.Merge(s, gd)

            t = dict()
            [t['char_inputs_in'], t['all_embs']] = \
                tf.import_graph_def(gd, {}, ['char_inputs_in:0', 'all_embs_out:0'], name='')

            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            sess.run(f'save/Assign', {'save/Const:0': ckpt_file})
            # The following was recovered from the graph structure, the first 62 assign modules
            # relate to the parameters of the char CNN.
            for i in range(1, 62):
                sess.run(f'save/Assign_{i}', {'save/Const:0': ckpt_file})

        return sess, t

    def encode(self, token: str) -> np.ndarray:
        if token in self.cnn_embs:
            return self.cnn_embs[token]

        input_dict = {
            self.cnn_t['char_inputs_in']: self.vocab.word_to_char_ids(token).reshape(
                [-1, 1, self.vocab.max_word_length])
        }
        emb = self.cnn_sess.run(self.cnn_t['all_embs'], input_dict)[0].astype(np.float32)
        self.cnn_embs[token] = emb

        return emb


class LSTM(nn.Module):
    def __init__(self,
                 ckpt_dir: str,
                 num_layers: int,
                 split_order: List[str],
                 forget_offset: int) -> None:
        super().__init__()

        print('Loading LSTM...')

        self.num_layers = num_layers
        self.split_order = split_order
        self.forget_offset = forget_offset

        # Projects hidden+input (2*1024) onto cell state dimension (8192)
        self.weight: ParameterDict = {}
        self.bias: ParameterDict = {}

        # Projects cell state dimension (8192) back to hidden dimension (1024)
        self.weight_P: ParameterDict = {}
        # The 3 peepholes are weighted by a diagonal matrix
        self.peepholes: PartialArrayDict = {}

        self._load_lstm(ckpt_dir)

    def _load_lstm(self, ckpt_dir: str) -> None:
        lstm_reader = NewCheckpointReader(os.path.join(ckpt_dir, 'ckpt-lstm'))

        for l in range(2):
            # Model weights are divided into 8 chunks
            # (32768, 2048)
            self.weight[l] = np.concatenate(
                [lstm_reader.get_tensor(f'lstm/lstm_{l}/W_{i}') for i in range(8)]
            ).astype(np.float32).T

            # (32768,)
            self.bias[l] = lstm_reader.get_tensor(f'lstm/lstm_{l}/B').astype(np.float32)

            # (8192, 1024)
            self.weight_P[l] = np.concatenate(
                [lstm_reader.get_tensor(f'lstm/lstm_{l}/W_P_{i}') for i in range(8)]
            ).astype(np.float32)

            for p in ['F', 'I', 'O']:
                self.peepholes[l, p.lower()] = \
                    lstm_reader.get_tensor(f'lstm/lstm_{l}/W_{p}_diag').astype(np.float32)

    def forward_step(self,
                     l: int,
                     inp: np.ndarray,
                     prev_hx: np.ndarray,
                     prev_cx: np.ndarray) -> NamedArrayDict:
        proj = self.weight[l] @ np.concatenate((inp, prev_hx)) + self.bias[l]
        split_proj = dict(zip(self.split_order, np.split(proj, 4)))

        f_g: np.ndarray = sigmoid(split_proj['f']
                                  + prev_cx*self.peepholes[l, 'f']
                                  + self.forget_offset)
        i_g: np.ndarray = sigmoid(split_proj['i']
                                  + prev_cx*self.peepholes[l, 'i']
                                  )
        c_tilde_g: np.ndarray = np.tanh(split_proj['g'])

        cx: np.ndarray = f_g * prev_cx + i_g * c_tilde_g

        o_g: np.ndarray = sigmoid(split_proj['o']
                                  + cx*self.peepholes[l, 'o']
                                  )

        hx: np.ndarray = (o_g * np.tanh(cx)) @ self.weight_P[l]

        return {
            'emb': inp,
            'hx': hx, 'cx': cx,
            'f_g': f_g, 'i_g': i_g, 'o_g': o_g, 'c_tilde_g': c_tilde_g
        }

    def forward(self,
                input_: np.ndarray,
                prev_activations: FullActivationDict) -> Tuple[np.ndarray, FullActivationDict]:
        # Iteratively compute and store intermediate rnn activations
        activations: FullActivationDict = {}

        for l in range(self.num_layers):
            prev_hx = prev_activations[l]['hx']
            prev_cx = prev_activations[l]['cx']
            activations[l] = self.forward_step(l, input_, prev_hx, prev_cx)
            input_ = activations[l]['hx']

        return input_, activations


class SoftMax:
    def __init__(self, vocab: C2I, full_vocab_path: str, ckpt_dir: str, hidden_size_h: int) -> None:
        print('Loading SoftMax...')
        self.decoder_w = np.zeros((len(vocab), hidden_size_h), dtype=np.float32)
        self.decoder_b = np.zeros(len(vocab), dtype=np.float32)

        self._load_softmax(vocab, full_vocab_path, ckpt_dir)

    def _load_softmax(self, vocab: C2I, full_vocab_path: str, ckpt_dir: str) -> None:
        with open(full_vocab_path) as f:
            full_vocab: List[str] = f.read().strip().split('\n')

        bias_reader = NewCheckpointReader(os.path.join(ckpt_dir, 'ckpt-softmax8'))
        full_bias = bias_reader.get_tensor('softmax/b')

        # SoftMax is chunked into 8 arrays of size 100000x1024
        for i in range(8):
            sm_reader = NewCheckpointReader(os.path.join(ckpt_dir, f'ckpt-softmax{i}'))

            sm_chunk = sm_reader.get_tensor(f'softmax/W_{i}').astype(np.float32)
            bias_chunk = full_bias[i:len(full_bias):8]
            vocab_chunk = full_vocab[i:len(full_bias):8]

            for j, w in enumerate(vocab_chunk):
                sm = sm_chunk[j]
                bias = bias_chunk[j]

                if w in vocab:
                    self.decoder_w[vocab[w]] = sm
                    self.decoder_b[vocab[w]] = bias

                if w == '</S>':
                    self.decoder_w[vocab[vocab.eos_token]] = sm
                    self.decoder_b[vocab[vocab.eos_token]] = bias
                elif w == '<UNK>':
                    self.decoder_w[vocab[vocab.unk_token]] = sm
                    self.decoder_b[vocab[vocab.unk_token]] = bias
