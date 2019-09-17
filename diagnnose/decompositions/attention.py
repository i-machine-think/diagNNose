import shutil
from typing import Any, Dict, List, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

import diagnnose.typedefs.config as config
from diagnnose.activations.activation_index import (
    activation_index_len,
    activation_index_to_iterable,
)
from diagnnose.decompositions import DecomposerFactory
from diagnnose.decompositions.base_decomposer import BaseDecomposer
from diagnnose.models.lm import LanguageModel
from diagnnose.typedefs.activations import ActivationIndex
from diagnnose.typedefs.corpus import Corpus
from diagnnose.utils.midpoint import MidPointNorm

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
            self.cd_config: Dict[str, Any] = {}
        else:
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
        extra_classes: Optional[List[int]] = None,
        arr_pickle: Optional[str] = None,
        save_arr_as: Optional[str] = None,
        save_plot_as: Optional[str] = None,
    ) -> Tensor:
        if arr_pickle is not None:
            arr: Tensor = torch.load(arr_pickle)
        else:
            arr = self.calc_by_sen_id(
                sen_ids,
                activations_dir=activations_dir,
                extra_classes=extra_classes,
                save_arr_as=save_arr_as,
            )

        if avg_decs:
            avg_arr = torch.mean(arr, dim=0)
            self.plot_attention(avg_arr, save_plot_as=save_plot_as)
        else:
            sen_ids = activation_index_to_iterable(sen_ids)
            batch_size = arr.size(0)
            y_idx = -len(extra_classes or []) - 1
            for i in range(batch_size):
                self.plot_config.update(
                    {
                        "xtext": self.corpus[sen_ids[i]].sen[1:],
                        "ytext": self.corpus[sen_ids[i]].sen[:y_idx],
                    }
                )
                self.plot_attention(arr[i])

        return arr

    def calc_by_sen_id(
        self,
        sen_ids: ActivationIndex,
        activations_dir: Optional[str] = None,
        extra_classes: Optional[List[int]] = None,
        save_arr_as: Optional[str] = None,
    ) -> Tensor:
        if isinstance(sen_ids, int):
            sen_ids = [sen_ids]
        classes = self._create_output_classes(sen_ids)

        if activations_dir is None:
            # We index by extraction position, so if N non-consecutive corpus items are being
            # decomposed we should pass a slice of N items along from here.
            sen_ids = slice(0, activation_index_len(sen_ids), 1)

        factory = DecomposerFactory(
            self.model,
            activations_dir or TMP_DIR,
            create_new_activations=(activations_dir is None),
            corpus=self.corpus,
            sen_ids=sen_ids,
        )

        decomposer = factory.create(
            sen_ids, classes=classes, extra_classes=extra_classes
        )

        arr = self.calc_attention(decomposer)

        if activations_dir is None:
            shutil.rmtree(TMP_DIR)

        if save_arr_as is not None:
            torch.save(arr, save_arr_as)

        return arr

    def calc_attention(self, decomposer: BaseDecomposer) -> Tensor:
        start_id = 0 if self.include_init else 1
        sen_len = int(decomposer.final_index[0])
        normalize = self.cd_config.pop("normalize", True)

        # Number of input features to be decomposed (init + w0 -- wn-1)
        ndecomp = sen_len - 1 + int(self.include_init)
        # Number of output classes (w1 -- wn)
        noutput = sen_len - 1 + len(decomposer.extra_classes)

        rel_scores = torch.zeros(
            (decomposer.batch_size, ndecomp, noutput), dtype=config.DTYPE
        )
        irrel_scores = torch.zeros(
            (decomposer.batch_size, ndecomp, noutput), dtype=config.DTYPE
        )

        for i in range(start_id, sen_len):
            # Note that this can slightly be improved, as we calculate the decoder score for each
            # output class at each time step. But for now this suffices.
            # Shape: (batch_size, ninput, noutput)
            decomposition = decomposer.decompose(i - 1, i, **self.cd_config)
            if not self.include_init:
                i -= 1
            rel_scores[:, i] = torch.diagonal(decomposition["relevant"], dim1=1, dim2=2)
            irrel_scores[:, i] = torch.diagonal(
                decomposition["irrelevant"], dim1=1, dim2=2
            )

        bias = decomposer.decoder_b.unsqueeze(1)
        if normalize:
            norm_scores = rel_scores / (rel_scores + irrel_scores + bias)
            return norm_scores

        self.cd_config["normalize"] = False
        return rel_scores

    def plot_attention(self, arr: Tensor, save_plot_as: Optional[str] = None) -> None:
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

        if "xtext" in self.plot_config:
            xtext = self.plot_config["xtext"]

            ax.set_xticks(range(len(xtext)))
            ax.set_xticklabels(xtext, rotation=35, ha="left", rotation_mode="anchor")
            ax.tick_params(axis="x", which="both", labelsize=32)
        else:
            ax.set_xticks([])

        if "ytext" in self.plot_config:
            ytext = self.plot_config["ytext"]
            if self.include_init:
                ytext = ["INIT"] + ytext

            ax.set_yticks(range(len(ytext)))
            ax.set_yticklabels(ytext)
            ax.tick_params(axis="y", which="both", labelsize=32)
        else:
            ax.set_yticks([])

        if self.plot_config.get("plot_values", True):
            fs = self.plot_config.get("value_font_size", 22)
            for (j, i), label in np.ndenumerate(arr):
                if label == 0.0:
                    continue
                if (cmin / 1.4) < label < (cmax / 1.4):
                    ax.text(
                        i,
                        j,
                        f"{label:.2f}",
                        ha="center",
                        va="center",
                        fontsize=fs,
                        color="black",
                    )
                else:
                    ax.text(
                        i,
                        j,
                        f"{label:.2f}",
                        ha="center",
                        va="center",
                        fontsize=fs,
                        color="white",
                    )

        ax.set_ylabel("Decomposed token", fontsize=28)
        ax.set_xlabel("Predicted class", fontsize=28)

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

        if "xtext_colors" in self.plot_config:
            for idx, color in self.plot_config["xtext_colors"]:
                ax.get_xticklabels()[idx].set_color(color)
        if "ytext_colors" in self.plot_config:
            for idx, color in self.plot_config["ytext_colors"]:
                ax.get_yticklabels()[idx].set_color(color)

        for patch_x, patch_y in self.plot_config.get("patches"):
            ax.add_patch(
                patches.Rectangle(
                    (patch_x - 0.47, patch_y - 0.48),
                    0.95,
                    0.94,
                    linewidth=4,
                    edgecolor="black",
                    facecolor="none",
                )
            )

        if self.plot_config.get("title", None) is not None:
            title = ax.set_title(self.plot_config["title"], fontsize=30)
            title.set_position([0.5, 1.2])

        if save_plot_as is not None:
            plt.savefig(save_plot_as, format="png")
        plt.show()

    def _create_output_classes(self, sen_ids: ActivationIndex) -> Tensor:
        classes: List[List[int]] = []
        for i, sen_id in enumerate(activation_index_to_iterable(sen_ids)):
            sen = self.corpus[sen_id].sen[1:]
            tokens = [self.corpus.vocab.stoi[w] for w in sen]
            classes.append(tokens)
            if i > 0:
                assert len(tokens) == len(
                    classes[0]
                ), "Unequal sentence lengths are not supported yet"

        return torch.tensor(classes)
