# diagNNose &middot; [![Build Status](https://travis-ci.com/i-machine-think/diagnnose.svg?branch=master)](https://travis-ci.com/i-machine-think/diagnnose) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

**Paper**: https://arxiv.org/abs/2011.06819

**Demo**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TzNU4l28NljkUcq_h1qbLizRAQuedvAZ?usp=sharing)

**Documentation**: https://diagnnose.readthedocs.io
#
This library contains a set of modules that can be used to analyse the activations of neural networks, with a focus on NLP architectures such as LSTMs and Transformers.
In particular, it contains functionality for :
- **Extracting** activations from different types of (language) models and providing quick access to these stored activations.
- Training **diagnostic classifiers** ([Hupkes et al., 2018](https://arxiv.org/pdf/1711.10203.pdf)) on extracted activations.
- Training **control tasks** ([Hewitt & Liang, 2019](https://arxiv.org/pdf/1909.03368.pdf)) parallel to these diagnostic classifiers.
- Performing *model-agnostic* **feature attributions** ([Murdoch et al., 2018](https://arxiv.org/pdf/1801.05453.pdf)) on a model.
- Running a broad linguistic suite of [**targeted syntactic evaluations**](https://github.com/i-machine-think/diagnnose/tree/master/diagnnose/downstream) on a language model.

:tada: `diagNNose` has been presented at [BlackboxNLP 2020](https://blackboxnlp.github.io/)! The paper can be found [here](https://www.aclweb.org/anthology/2020.blackboxnlp-1.32.pdf).

**Documentation** can be found at [diagnnose.readthedocs.io](https://diagnnose.readthedocs.io).

Our library is officially registered with pip and can be installed by running `pip install diagnnose`.
The preferred version of Python is ≥3.6. The required packages are stated in [`requirements.txt`](https://github.com/i-machine-think/diagnnose/blob/master/requirements.txt).
#
### Quick Tour
The workflow of `diagNNose` is divided into several building blocks, that can be combined for various experiments.

We provide a few examples that demonstrate the library. 
An interactive and more extensive interface for these scripts is also provided in the form of a notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1TzNU4l28NljkUcq_h1qbLizRAQuedvAZ?usp=sharing)

#### Activation Extraction
The activations of a model can be extracted using an `Extractor` that takes care of batching and selecting activations of interest.

Fine-grained activation selection is possible by defining a `selection_func`, that selects an activation based on the current sentence index and corpus item.

```python
from torchtext.data import Example

from diagnnose.config import create_config_dict
from diagnnose.corpus import Corpus
from diagnnose.extract import Extractor
from diagnnose.models import LanguageModel, import_model
from diagnnose.tokenizer.create import create_tokenizer

if __name__ == "__main__":
    config_dict = create_config_dict()

    tokenizer = create_tokenizer(**config_dict["tokenizer"])
    corpus: Corpus = Corpus.create(tokenizer=tokenizer, **config_dict["corpus"])
    model: LanguageModel = import_model(**config_dict["model"])

    def selection_func(w_idx: int, item: Example) -> bool:
        return w_idx == item.extraction_idx

    extractor = Extractor(
        model, corpus, selection_func=selection_func, **config_dict["extract"]
    )
    activation_reader = extractor.extract()
```

#
#### Research using `diagNNose`
- Jumelet, Zuidema & Hupkes (2019): _[Analysing Neural Language Models: Contextual Decomposition Reveals Default Reasoning in Number and Gender Assignment](https://arxiv.org/pdf/1909.08975.pdf)_

#### Citing
If you intend on using `diagNNose` for your research, please cite us as follows. Feel free to reach out, we'd love to help!
```
@inproceedings{jumelet-2020-diagnnose,
    title = "diag{NN}ose: A Library for Neural Activation Analysis",
    author = "Jumelet, Jaap",
    booktitle = "Proceedings of the Third BlackboxNLP Workshop on Analyzing and Interpreting Neural Networks for NLP",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.blackboxnlp-1.32",
    pages = "342--350",
}
```
