import shutil
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from diagnnose.decompositions import DecomposerFactory
from diagnnose.decompositions.base_decomposer import BaseDecomposer
from diagnnose.extractors.base_extractor import Extractor
from diagnnose.typedefs.activations import ActivationNames
from diagnnose.typedefs.corpus import Corpus
from diagnnose.typedefs.models import LanguageModel
from diagnnose.utils.midpoint import MidPointNorm
from diagnnose.utils.misc import suppress_print

plt.rcParams["figure.figsize"] = 15, 10

TMP_DIR = "tmp"


class CDAttention:
    def __init__(
        self,
        model: LanguageModel,
        include_init: bool = True,
        cd_config: Optional[Dict[str, Any]] = None,
        plot_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model = model
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

    def calc_by_sen_id(
        self, corpus: Corpus, sen_id: int, activations_dir: Optional[str] = None
    ) -> np.ndarray:
        sen: List[str] = corpus[sen_id].sen
        classes: List[int] = [corpus.vocab.stoi[w] for w in sen]

        factory = self._create_factory(corpus, sen_id, activations_dir)
        if activations_dir is None:
            sen_id = 0

        decomposer = factory.create(sen_id, classes=classes)

        arr = self.calc_attention(decomposer, len(sen))

        if activations_dir is None:
            shutil.rmtree(TMP_DIR)

        return arr

    def plot_by_sen_id(
        self, corpus: Corpus, sen_id: int, activations_dir: Optional[str] = None
    ) -> np.ndarray:
        sen: List[str] = corpus[sen_id].sen

        arr = self.calc_by_sen_id(corpus, sen_id, activations_dir=activations_dir)

        self.plot_attention(arr, sen)

        return arr

    def calc_attention(
        self, decomposer: BaseDecomposer, sen_len: int, normalize: bool = True
    ) -> np.ndarray:
        start_id = 0 if self.include_init else 1
        end_id = sen_len

        # Number of input features to be decomposed (init + w0 -- wn-1)
        ndecomp = sen_len - 1 + int(self.include_init)
        # Number of output classes (w1 -- wn)
        noutput = sen_len - 1

        rel_scores = np.zeros((ndecomp, noutput))
        irrel_scores = np.zeros((ndecomp, noutput))

        for i in range(start_id, end_id):
            decomposition = decomposer.decompose(
                i - 1, i, only_return_dec=True, **self.cd_config
            )

            if not self.include_init:
                i -= 1

            for j in range(1, end_id):
                rel_scores[i, j - 1] = (
                    decomposition["rel_h"][j - 1] @ decomposer.decoder_w[j].T
                )
                irrel_scores[i, j - 1] = (
                    decomposition["irrel_h"][j - 1] @ decomposer.decoder_w[j].T
                )

        if normalize:
            return rel_scores / (rel_scores + irrel_scores)

        return rel_scores

    def plot_attention(self, arr: np.ndarray, sen: List[str]) -> None:
        arr_mask = np.ma.masked_array(arr, mask=(arr != 0.0))
        arr = np.ma.masked_array(arr, mask=(arr == 0))

        xtext = sen[1:]
        ytext = sen[:-1]
        if self.include_init:
            ytext = ["INIT"] + ytext

        if self.plot_config.get("clim", None) is not None:
            clim = self.plot_config["clim"]
            cmin = clim[0]
            cmax = clim[1]
        else:
            cmin = min(np.min(arr), 0)
            cmax = np.max(arr)
        cmid = 0 if cmin < 0 < cmax else np.min(arr)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.imshow(
            arr,
            cmap=self.plot_config.get("cmap", "coolwarm"),
            clim=(cmin, cmax),
            norm=MidPointNorm(cmid, vmin=cmin, vmax=cmax),
        )
        ax.imshow(arr_mask, cmap="gist_yarg")

        ax.set_xticks(range(len(xtext)))
        ax.set_xticklabels(xtext, rotation=35, ha="left", rotation_mode="anchor")
        ax.set_yticks(range(len(ytext)))
        ax.set_yticklabels(ytext)

        ax.tick_params(axis="y", which="both", labelsize=30)
        ax.tick_params(axis="x", which="both", labelsize=26)

        if self.plot_config.get("plot_values", True):
            for (j, i), label in np.ndenumerate(arr):
                if label == 0.0:
                    continue
                beta = np.round(label, 2)
                if (cmin / 1.5) < beta < (cmax / 1.5):
                    ax.text(
                        i, j, beta, ha="center", va="center", fontsize=22, color="black"
                    )
                else:
                    ax.text(
                        i, j, beta, ha="center", va="center", fontsize=22, color="white"
                    )

        ax.set_ylabel("Decomposed token", fontsize=24)
        ax.set_xlabel("Predicted class", fontsize=24)

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

        if self.plot_config.get("title", None) is not None:
            ax.set_title(self.plot_config["title"], fontsize=14)

        plt.show()

    def _create_factory(
        self, corpus: Corpus, sen_id: int, activations_dir: Optional[str]
    ) -> DecomposerFactory:
        if activations_dir is None:
            activations_dir = TMP_DIR
            activation_names = self._get_activation_names()

            all_examples = list(corpus.examples)
            corpus.examples = [corpus.examples[sen_id]]  # discard all other items
            extractor = Extractor(self.model, corpus, activations_dir)

            self._extract(extractor, activation_names)
            corpus.examples = all_examples

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
