import shutil
from typing import Any, Dict, List, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from diagnnose.activations.activation_index import (
    activation_index_len,
    activation_index_to_iterable,
)
from diagnnose.decompositions import DecomposerFactory
from diagnnose.decompositions.base_decomposer import BaseDecomposer
from diagnnose.extractors.base_extractor import Extractor
from diagnnose.models.lm import LanguageModel
from diagnnose.typedefs.activations import ActivationIndex, ActivationNames
from diagnnose.typedefs.corpus import Corpus
from diagnnose.utils.midpoint import MidPointNorm
from diagnnose.utils.misc import suppress_print

plt.rcParams["figure.figsize"] = 15, 10

TMP_DIR = "tmp"


class CDAttention:
    def __init__(
        self,
        model: LanguageModel,
        corpus: Corpus,
        include_init: bool = True,
        cd_config: Optional[Dict[str, Any]] = None,
        plot_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
        self.corpus = corpus
        self.include_init = include_init

        if cd_config is None:
            self.decomposer = "ContextualDecomposer"
            self.cd_config: Dict[str, Any] = {}
        else:
            self.decomposer = cd_config.pop("decomposer", "ContextualDecomposer")
            self.cd_config = cd_config

        if plot_config is None:
            self.plot_config: Dict[str, Any] = {}
        else:
            self.plot_config = plot_config

    def plot_by_sen_id(
        self,
        sen_ids: ActivationIndex,
        activations_dir: Optional[str] = None,
        avg_decs: bool = False,
        arr_pickle: Optional[str] = None,
        save_arr_as: Optional[str] = None,
        save_plot_as: Optional[str] = None,
    ) -> Tensor:
        if arr_pickle is not None:
            arr: Tensor = torch.load(arr_pickle)
        else:
            arr = self.calc_by_sen_id(
                sen_ids, activations_dir=activations_dir, save_arr_as=save_arr_as
            )

        if avg_decs:
            avg_arr = torch.mean(arr, dim=0)
            sen = self.plot_config.get("sen", None)
            self.plot_attention(avg_arr, sen=sen, save_plot_as=save_plot_as)
        else:
            sen_ids = activation_index_to_iterable(sen_ids)
            batch_size = arr.size(0)
            for i in range(batch_size):
                sen_id = sen_ids[i]  # mypy error fixable with PEP 544
                self.plot_attention(arr[i], sen=self.corpus[sen_id].sen)

        return arr

    def calc_by_sen_id(
        self,
        sen_ids: ActivationIndex,
        activations_dir: Optional[str] = None,
        save_arr_as: Optional[str] = None,
    ) -> Tensor:
        if isinstance(sen_ids, int):
            sen_ids = [sen_ids]
        classes = self._create_output_classes(sen_ids)
        factory = self._create_factory(sen_ids, activations_dir)
        if activations_dir is None:
            # We index by extraction position, so if N non-consecutive corpus items are being
            # decomposed we should pass a slice of N items along from here.
            sen_ids = slice(0, activation_index_len(sen_ids), 1)

        decomposer = factory.create(sen_ids, classes=classes)

        arr = self.calc_attention(decomposer)

        if activations_dir is None:
            shutil.rmtree(TMP_DIR)

        if save_arr_as is not None:
            torch.save(arr, save_arr_as)

        return arr

    def calc_attention(
        self, decomposer: BaseDecomposer
    ) -> Tensor:
        start_id = 0 if self.include_init else 1
        sen_len = int(decomposer.final_index[0])
        normalize = self.cd_config.pop('normalize', True)

        # Number of input features to be decomposed (init + w0 -- wn-1)
        ndecomp = sen_len - 1 + int(self.include_init) + 3
        # Number of output classes (w1 -- wn)
        noutput = sen_len - 1

        rel_scores = torch.zeros((decomposer.batch_size, ndecomp, noutput))
        irrel_scores = torch.zeros((decomposer.batch_size, ndecomp, noutput))

        for i in range(start_id, sen_len):
            # Note that this can slightly be improved, as we calculate the decoder score for each
            # output class at each time step. But for now this suffices.
            # Shape: (batch_size, ninput, noutput
            decomposition = decomposer.decompose(i - 1, i, **self.cd_config)
            if not self.include_init:
                i -= 1
            rel_scores[:, i] = torch.diagonal(
                decomposition["relevant"][:, :, 1:], dim1=1, dim2=2
            )
            irrel_scores[:, i] = torch.diagonal(
                decomposition["irrelevant"][:, :, 1:], dim1=1, dim2=2
            )

        bias = decomposer.decoder_b[:, 1:].unsqueeze(1)
        if normalize:
            norm_scores = rel_scores / (rel_scores + irrel_scores + bias)
            return norm_scores

        rel_scores[:, -3] = torch.sum(rel_scores, dim=1)
        rel_scores[:, -2] = rel_scores[:, 0] + irrel_scores[:, 0]
        rel_scores[:, -1] = rel_scores[:, -3] / rel_scores[:, -2]

        for i in range(rel_scores.size(0)):
            rel_scores[i, :-3] /= rel_scores[i, -3]

        self.cd_config["normalize"] = False
        return rel_scores

    def plot_attention(
        self,
        arr: Tensor,
        sen: Optional[List[str]] = None,
        save_plot_as: Optional[str] = None,
    ) -> None:
        arr = arr.numpy()
        arr_mask = np.ma.masked_array(arr, mask=(arr != 0.0))
        arr = np.ma.masked_array(arr, mask=(arr == 0))

        # clim can be provided as [cmin, cmax], or [cmin, cmid, cmax]
        if self.plot_config.get("clim", None) is not None:
            clim = self.plot_config["clim"]
            cmin = clim[0]
            cmax = clim[1] if len(clim) == 2 else clim[2]
        else:
            cmin = min(np.min(arr), 0)
            cmax = np.max(arr)
        if len(self.plot_config.get("clim", [])) == 3:
            cmid = self.plot_config["clim"][1]
        else:
            cmid = 0 if cmin < 0 < cmax else np.min(arr) + 1e-8

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.imshow(
            arr,
            cmap=self.plot_config.get("cmap", "coolwarm"),
            clim=(cmin, cmax),
            norm=MidPointNorm(cmid, vmin=cmin, vmax=cmax),
        )
        ax.imshow(arr_mask, cmap="gist_yarg")

        if sen is not None:
            xtext = sen[1:]
            ytext = sen[:-1]
            if self.include_init:
                ytext = ["INIT"] + ytext

            ax.set_xticks(range(len(xtext)))
            ax.set_xticklabels(xtext, rotation=35, ha="left", rotation_mode="anchor")
            ax.set_yticks(range(len(ytext)))
            ax.set_yticklabels(ytext)

            ax.tick_params(axis="y", which="both", labelsize=30)
            ax.tick_params(axis="x", which="both", labelsize=26)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        if self.plot_config.get("plot_values", True):
            fs = self.plot_config.get("value_font_size", 22)
            for (j, i), label in np.ndenumerate(arr):
                if label == 0.0:
                    continue
                beta = np.round(label, 2)
                if (cmin / 1.5) < beta < (cmax / 1.3):
                    ax.text(
                        i, j, beta, ha="center", va="center", fontsize=fs, color="black"
                    )
                else:
                    ax.text(
                        i, j, beta, ha="center", va="center", fontsize=fs, color="white"
                    )

        ax.set_ylabel("Decomposed token", fontsize=24)
        ax.set_xlabel("Predicted class", fontsize=24)

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

        if self.plot_config.get("title", None) is not None:
            title = ax.set_title(self.plot_config["title"], fontsize=30)
            title.set_position([0.5, 1.2])

        if save_plot_as is not None:
            plt.savefig(save_plot_as, format="png")
        plt.show()

    def _create_output_classes(self, sen_ids: ActivationIndex) -> Tensor:
        classes: List[List[int]] = []
        for i, sen_id in enumerate(activation_index_to_iterable(sen_ids)):
            sen = self.corpus[sen_id].sen
            classes.append([self.corpus.vocab.stoi[w] for w in sen])
            if i > 0:
                assert len(sen) == len(
                    classes[0]
                ), "Unequal sentence lengths are not supported yet"

        return torch.tensor(classes)

    def _create_factory(
        self, sen_ids: ActivationIndex, activations_dir: Optional[str]
    ) -> DecomposerFactory:
        if activations_dir is None:
            activations_dir = TMP_DIR
            activation_names = self._get_activation_names()

            all_examples = list(self.corpus.examples)  # create copy of full corpus
            self.corpus.examples = [
                self.corpus.examples[idx]
                for idx in activation_index_to_iterable(sen_ids)
            ]  # discard all other items
            extractor = Extractor(self.model, self.corpus, activations_dir)

            self._extract(extractor, activation_names)
            self.corpus.examples = all_examples  # restore initial corpus

        factory = DecomposerFactory(
            self.model, activations_dir, decomposer=self.decomposer
        )

        return factory

    def _get_activation_names(self) -> ActivationNames:
        activation_names: ActivationNames = []

        if self.decomposer == "CellDecomposer":
            activation_names.extend(
                [
                    (self.model.num_layers - 1, name)
                    for name in ["f_g", "o_g", "hx", "cx", "icx", "0cx"]
                ]
            )
        else:
            for l in range(self.model.num_layers):
                activation_names.extend([(l, "cx"), (l, "hx")])
            activation_names.append((0, "emb"))

        return activation_names

    @staticmethod
    @suppress_print
    def _extract(extractor: Extractor, activation_names: ActivationNames) -> None:
        extractor.extract(activation_names, dynamic_dumping=False)
