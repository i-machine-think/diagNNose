import os
from itertools import product
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn.functional import log_softmax
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence
from transformers import PreTrainedTokenizer

from diagnnose.activations.selection_funcs import final_sen_token
from diagnnose.attribute import ShapleyTensor
from diagnnose.corpus import Corpus
from diagnnose.extract import Extractor
from diagnnose.models import LanguageModel
from diagnnose.typedefs.activations import (
    ActivationDict,
    ActivationName,
    ActivationNames,
)
from diagnnose.utils import __file__ as diagnnose_utils_init
from diagnnose.utils.misc import suppress_print
from diagnnose.utils.pickle import load_pickle


class RecurrentLM(LanguageModel):
    """Base class for RNN LM with intermediate activations.

    This class contains all the base logic (including forward passes)
    for LSTM-type LMs, except for loading in the weights of a specific
    model.
    """

    is_causal: bool = True
    forget_offset: int = 0
    ih_concat_order: List[str] = ["h", "i"]
    split_order: List[str]
    use_char_embs: bool = False
    use_peepholes: bool = False
    init_states: ActivationDict = {}

    def __init__(self, device: str = "cpu"):
        super().__init__(device)

        # layer index -> layer weights
        self.weight: Dict[int, Tensor] = {}
        self.bias: Dict[int, Tensor] = {}

        # Projects cell state dimension (8192) back to hidden dimension (1024)
        self.weight_P: Dict[int, Tensor] = {}
        # The 3 peepholes are weighted by a diagonal matrix
        self.peepholes: ActivationDict = {}

        self.decoder_w: Optional[Tensor] = None
        self.decoder_b: Optional[Tensor] = None

    def create_inputs_embeds(self, input_ids: Tensor) -> Tensor:
        return self.word_embeddings[input_ids]

    def decode(self, hidden_state: Tensor) -> Tensor:
        return hidden_state @ self.decoder_w.t() + self.decoder_b

    @property
    def num_layers(self) -> int:
        return max(layer for layer, _name in self.sizes) + 1

    @property
    def top_layer(self) -> int:
        return self.num_layers - 1

    @property
    def output_size(self) -> int:
        return self.sizes[self.top_layer, "hx"]

    def nhid(self, activation_name: ActivationName) -> int:
        """Returns number of hidden units for a (layer, name) tuple.

        If `name` != emb/hx/cx returns the size of (layer, `cx`).
        """
        layer, name = activation_name

        return self.sizes.get((layer, name), self.sizes[layer, "cx"])

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

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Union[Tensor, ShapleyTensor]] = None,
        input_lengths: Optional[Tensor] = None,
        calc_causal_lm_probs: bool = False,
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

        inputs_embeds = inputs_embeds.to(self.device)

        iterator, unsorted_indices = self._create_iterator(inputs_embeds, input_lengths)

        all_activations = self._init_activations(inputs_embeds, compute_out)
        cur_activations = self.init_hidden(inputs_embeds.size(0))

        for w_idx, input_ in enumerate(iterator):
            num_input = input_.shape[0]
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

        if calc_causal_lm_probs:
            output_ids = input_ids[:, 1:].unsqueeze(-1)
            logits = all_activations[self.top_layer, "out"]
            probs = log_softmax(logits[:, :-1], dim=-1)
            all_activations[self.top_layer, "out"] = torch.gather(probs, -1, output_ids)

        if only_return_top_embs and compute_out:
            return all_activations[self.top_layer, "out"]
        elif only_return_top_embs:
            return all_activations[self.top_layer, "hx"]

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

        if self.use_peepholes:
            split_proj["f"] += prev_cx * self.peepholes[layer, "f"]
            split_proj["i"] += prev_cx * self.peepholes[layer, "i"]

        # Shapes: (bsz, nhid_c)
        f_g = torch.sigmoid(split_proj["f"])
        i_g = torch.sigmoid(split_proj["i"])
        c_tilde_g = torch.tanh(split_proj["g"])

        cx = f_g * prev_cx + i_g * c_tilde_g

        if self.use_peepholes:
            split_proj["o"] += cx * self.peepholes[layer, "o"]
        o_g = torch.sigmoid(split_proj["o"])
        hx = o_g * torch.tanh(cx)

        if self.sizes[layer, "hx"] != self.sizes[layer, "cx"]:
            hx = hx @ self.weight_P[layer]

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

    @staticmethod
    def _create_iterator(
        inputs_embeds: Tensor, input_lengths: Optional[Tensor]
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
            batch_size = inputs_embeds.shape[0]
            input_lengths = torch.tensor(batch_size * [inputs_embeds.shape[1]])

        packed_batch: PackedSequence = pack_padded_sequence(
            inputs_embeds,
            lengths=input_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
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
                all_activations[a_name] = type(inputs_embeds)(activations)

        return all_activations

    def init_hidden(self, batch_size: int) -> ActivationDict:
        """Creates a batch of initial states.

        Parameters
        ----------
        batch_size : int
            Size of batch for which states are created.

        Returns
        -------
        init_states : ActivationTensors
            Dictionary mapping hidden and cell state to init tensors.
        """
        batch_init_states: ActivationDict = {}

        for layer in range(self.num_layers):
            for hc in ["hx", "cx"]:
                # Shape: (batch_size, nhid)
                batched_state = self.init_states[layer, hc].repeat(batch_size, 1)
                batch_init_states[layer, hc] = batched_state

        return batch_init_states

    def final_hidden(self, hidden: ActivationDict) -> Tensor:
        """Returns the final hidden state.

        Parameters
        ----------
        hidden : ActivationTensors
            Dictionary of extracted activations.

        Returns
        -------
        final_hidden : Tensor
            Tensor of the final hidden state.
        """
        return hidden[self.top_layer, "hx"].squeeze()

    def set_init_states(
        self,
        pickle_path: Optional[str] = None,
        corpus_path: Optional[str] = None,
        use_default: bool = False,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        save_init_states_to: Optional[str] = None,
    ) -> None:
        """Set up the initial LM states.

        If no path is provided 0-valued embeddings will be used.
        Note that the loaded init should provide tensors for `hx`
        and `cx` in all layers of the LM.

        Note that `init_states_pickle` takes precedence over
        `init_states_corpus` in case both are provided.

        Parameters
        ----------
        pickle_path : str, optional
            Path to pickled file with initial lstm states. If not
            provided zero-valued init states will be created.
        corpus_path : str, optional
            Path to corpus of which the final hidden state will be used
            as initial states.
        use_default : bool
            Toggle to use the default initial sentence `. <eos>`.
        tokenizer : PreTrainedTokenizer, optional
            Tokenizer that must be provided when creating the init
            states from a corpus.
        save_init_states_to : str, optional
            Path to which the newly computed init_states will be saved.
            If not provided these states won't be dumped.

        Returns
        -------
        init_states : ActivationTensors
            ActivationTensors containing the init states for each layer.
        """
        if use_default:
            diagnnose_utils_dir = os.path.dirname(diagnnose_utils_init)
            corpus_path = os.path.join(diagnnose_utils_dir, "init_sentence.txt")

        if pickle_path is not None:
            init_states = self._create_init_states_from_pickle(pickle_path)
        elif corpus_path is not None:
            init_states = self._create_init_states_from_corpus(
                corpus_path, tokenizer, save_init_states_to
            )
        else:
            init_states = self._create_zero_states()

        self.init_states = init_states

    def _create_zero_states(self) -> ActivationDict:
        """Zero-initialized states if no init state is provided.

        Returns
        -------
        init_states : ActivationTensors
            Dictionary mapping (layer, name) tuple to zero-tensor.
        """
        init_states: ActivationDict = {
            a_name: torch.zeros((1, self.nhid(a_name)), device=self.device)
            for a_name in product(range(self.num_layers), ["cx", "hx"])
        }

        return init_states

    @suppress_print
    def _create_init_states_from_corpus(
        self,
        init_states_corpus: str,
        tokenizer: PreTrainedTokenizer,
        save_init_states_to: Optional[str] = None,
    ) -> ActivationDict:
        assert (
            tokenizer is not None
        ), "Tokenizer must be provided when creating init states from corpus"

        corpus: Corpus = Corpus.create(init_states_corpus, tokenizer=tokenizer)

        activation_names: ActivationNames = [
            (layer, name) for layer in range(self.num_layers) for name in ["hx", "cx"]
        ]

        extractor = Extractor(
            self,
            corpus,
            activation_names,
            activations_dir=save_init_states_to,
            selection_func=final_sen_token,
        )
        init_states = extractor.extract().activation_dict

        return init_states

    def _create_init_states_from_pickle(self, pickle_path: str) -> ActivationDict:
        init_states: ActivationDict = load_pickle(pickle_path)

        self._validate_init_states_from_pickle(init_states)

        return init_states

    def _validate_init_states_from_pickle(self, init_states: ActivationDict) -> None:
        num_init_layers = max(layer for layer, _name in init_states)
        assert (
            num_init_layers == self.num_layers
        ), "Number of initial layers not correct"

        for (layer, name), size in self.sizes.items():
            if name in ["hx", "cx"]:
                assert (
                    layer,
                    name,
                ) in init_states.keys(), (
                    f"Activation {layer},{name} is not found in init states"
                )

                init_size = init_states[layer, name].size(1)
                assert init_size == size, (
                    f"Initial activation size for {name} is incorrect: "
                    f"{name}: {init_size}, should be {size}"
                )
