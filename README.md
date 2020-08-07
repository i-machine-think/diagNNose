# diagnnose &middot; [![Build Status](https://travis-ci.com/i-machine-think/diagnnose.svg?branch=master)](https://travis-ci.com/i-machine-think/diagnnose) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

:construction: **Currently undergoing construction, with a planned release in early September.** :construction:

---

This library contains a set of modules that can be used to analyse recurrent neural networks.
In particular, it contains functionality for :
- **Extracting** activations from different types of (language) models and providing quick access to these stored activations.
- Running **diagnostic classifiers** ([Hupkes et al., 2018](https://arxiv.org/pdf/1711.10203.pdf)) on extracted activations.
- Running **control tasks** ([Hewitt & Liang, 2019](https://arxiv.org/pdf/1909.03368.pdf)) parallel to these diagnostic classifiers.
- Performing **contextual decomposition** ([Murdoch et al., 2018](https://arxiv.org/pdf/1801.05453.pdf)) on a model.
- Running a broad linguistic [**downstream task suite**](https://github.com/i-machine-think/diagnnose/tree/master/diagnnose/downstream) on a language model.

**Documentation** can be found at [diagnnose.readthedocs.io](https://diagnnose.readthedocs.io).

Our library is officially registered with pip and can be installed by running `pip install diagnnose`.
The preferred version of Python is ≥3.7. The required packages are stated in [`requirements.txt`](https://github.com/i-machine-think/diagnnose/blob/master/requirements.txt).

### Research using `diagnnose`
- Jumelet, Zuidema & Hupkes (2019): _[Analysing Neural Language Models: Contextual Decomposition Reveals Default Reasoning in Number and Gender Assignment](https://arxiv.org/pdf/1909.08975.pdf)_

#### Citing
If you intend on using `diagnnose` for your research, please cite us as follows. Feel free to reach out, we'd love to help!
```
@article{diagnnose,
    title={diagnnose: A Neural Net Analysis Library},
    DOI={10.5281/zenodo.3445477},
    publisher={Zenodo},
    author={Jaap Jumelet and Dieuwke Hupkes},
    year={2019},
}
```
