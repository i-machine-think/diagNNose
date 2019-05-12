import os
import sys
from typing import Any, Tuple

import numpy as np
import tensorflow as tf
import torch
from google.protobuf import text_format
from overrides import overrides
from scipy.special import expit as sigmoid
from tensorflow.python import pywrap_tensorflow
from torch import Tensor

from diagnnose.typedefs.activations import NamedArrayDict, FullActivationDict, ParameterDict
from diagnnose.utils.w2i import C2I, create_vocab_from_path

from .language_model import LanguageModel


class GoogleLM(LanguageModel):
    def __init__(self, vocab_path: str, pbtxt_path: str, ckpt_dir: str) -> None:
        super().__init__()

        self.num_layers = 2
        self.hidden_size_c = 8192
        self.hidden_size_h = 1024
        self.split_order = ['f', 'i', 'o', 'g']
        self.array_type = 'numpy'

        self.c2i = C2I(create_vocab_from_path(vocab_path))

        self.cnn_sess, self.cnn_t = self._load_char_cnn(pbtxt_path, ckpt_dir)
        self.cnn_embs: NamedArrayDict = {}

        print('Loading LSTM...')
        lstm_reader = pywrap_tensorflow.NewCheckpointReader(os.path.join(ckpt_dir, 'ckpt-lstm'))

        # Projects hidden+input (2*1024) onto cell state dimension (8192)
        self.weight: ParameterDict = {}
        self.bias: ParameterDict = {}

        # Projects cell state dimension (8192) back to hidden dimension (1024)
        self.weight_P: ParameterDict = {}
        # The 3 peepholes are weighted by a diagonal matrix
        self.peepholes: ParameterDict = {0: {}, 1: {}}

        for l in range(self.num_layers):
            # Model weights are divided into 8 chunks
            # (2048, 32768)
            self.weight[l] = np.concatenate(
                [lstm_reader.get_tensor(f'lstm/lstm_{l}/W_{i}') for i in range(8)]
            )
            # (32768,)
            self.bias[l] = lstm_reader.get_tensor(f'lstm/lstm_{l}/B')

            # (8192, 1024)
            self.weight_P[l] = np.concatenate(
                [lstm_reader.get_tensor(f'lstm/lstm_{l}/W_P_{i}') for i in range(8)]
            )
            for p in ['F', 'I', 'O']:
                self.peepholes[l][p] = lstm_reader.get_tensor(f'lstm/lstm_{l}/W_{p}_diag')

        print('Model initialisation finished.')

    @staticmethod
    def _load_char_cnn(pbtxt_path: str, ckpt_dir: str) -> Any:
        print('Loading char CNN...')
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
            self.cnn_t['char_inputs_in']: self.c2i.word_to_char_ids(token).reshape(
                [-1, 1, self.c2i.max_word_length])
        }
        emb = self.cnn_sess.run(self.cnn_t['all_embs'], input_dict)[0]
        self.cnn_embs[token] = emb

        return emb

    def forward_step(self,
                     l: int,
                     inp: np.ndarray,
                     prev_hx: np.ndarray,
                     prev_cx: np.ndarray) -> NamedArrayDict:
        proj = np.concatenate((prev_hx, inp)) @ self.weight[l] + self.bias[l]
        split_proj = dict(zip(self.split_order, np.split(proj, 4)))

        f_g: np.ndarray = sigmoid(split_proj['f'] + prev_cx*self.peepholes[l]['F'] + 1)
        i_g: np.ndarray = sigmoid(split_proj['i'] + prev_cx*self.peepholes[l]['I'])
        c_tilde_g: np.ndarray = np.tanh(split_proj['g'])

        cx: np.ndarray = f_g * prev_cx + i_g * c_tilde_g

        o_g: np.ndarray = sigmoid(split_proj['o'] + cx*self.peepholes[l]['O'])

        hx: np.ndarray = (o_g * np.tanh(cx)) @ self.weight_P[l]

        return {
            'emb': inp,
            'hx': hx, 'cx': cx,
            'f_g': f_g, 'i_g': i_g, 'o_g': o_g, 'c_tilde_g': c_tilde_g
        }

    @overrides
    def forward(self,
                token: str,
                prev_activations: FullActivationDict) -> Tuple[np.ndarray, FullActivationDict]:
        # Look up the embeddings of the input words
        input_ = self.encode(token)

        # Iteratively compute and store intermediate rnn activations
        activations: FullActivationDict = {}
        for l in range(self.num_layers):
            prev_hx = prev_activations[l]['hx']
            prev_cx = prev_activations[l]['cx']
            activations[l] = self.forward_step(l, input_, prev_hx, prev_cx)
            input_ = activations[l]['hx']

        # out: Tensor = self.w_decoder @ input_ + self.b_decoder

        return input_, activations
