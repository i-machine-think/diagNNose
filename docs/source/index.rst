diagnnose documentation
=======================

``diagnnose`` is a Python library for analysing the activations of deep neural networks.
``diagnnose``'s main focus currently lies on techniques that aid in uncovering linguistic knowledge that is encoded within a model’s representations.
The library provides abstractions that allow re-current models to be investigated in the sameway as Transformer models, in a modular fashion.
It contains an extensive **activation extraction module** that allows for the extraction of (intermediate) model activations on a corpus.
The analysis techniques that are currently implemented include:

* Targeted syntactic evaluation tasks, such as those of Linzen et al. (2016) andMarvin and Linzen (2018).
* Probing with diagnostic classifiers (`Hupkes et al., 2018 <https://arxiv.org/pdf/1711.10203.pdf>`_), and control tasks (`Hewitt & Liang, 2019 <https://arxiv.org/pdf/1909.03368.pdf>`_).
* Feature attributions that retrieve a feature’s contribution to a model prediction (`Murdoch et al., 2018 <https://arxiv.org/pdf/1801.05453.pdf>`_). Our implementation is model-agnostic, which means that any type of model architecture can be explained by it.

The library is structured into the following modules:

.. glossary::

    models
        We provide an abstraction over language models, enabling recurrent and Transformer models to derive from the same interface.
        Importing pre-trained Transformer models is done via the ``transformers`` library.
        For recurrent models we provide a wrapper that enables access to intermediate activations, including gate activations.
        We also provide functionality that allows to set the initial hidden states of recurrent LMs, based on a sentence or corpus.

    corpus
        Corpora are imported as a ``Dataset`` from the ``torchtext`` package.
        A Corpus can be transformed into an iterator for processing.
        Tokenization is performed for both model types using the ``transformers`` tokenizers, which allows tokenization to be done in a traditional token-per-token fashion, or based on byte pair encodings.

    extract
        Central to most of the analysis modules is the extraction of activations.
        We provide an ``Extractor`` class that can extract the activations of a model given a corpus.
        Thanks to our model wrappers activation extraction is not restricted to just the top layer of a model; intermediate (gate) activations can be extracted as well.
        To facilitate the extraction of larger corpora with limited computational resources, activations can be dumped dynamically to disk.

    activations
        Extracted activations can easily be retrieved using a ``ActivationReader``, that provides access to the activations corresponding to a specific subset of corpus sentences.
        We also provide functionality for extracting only a specific subset of activations, based on sentence and token information.
        This way it is possible, for instance, to only extract the activations at the position of tokens of particular interest.

    config
        The pipeline of ``diagnnose`` is configuration-driven.
        Configuration is defined in JSON format, but individual attributes can also be set from the command line directly.

    syntax
        The library provides functionality for a large suite of targeted syntactic evaluation tasks.
        Currently we provide support for the following tasks:

        * The subject-verb agreement corpus of `Linzen et al., (2016) <https://arxiv.org/abs/1611.01368>`_, for which we also provide more fine-grained attractor conditions;
        * The wide range of linguistic expressions of `Marvin and Linzen, (2018) <https://arxiv.org/abs/1808.09031>`_;
        * The subject-verb agreement tasks of `Lakretz et al., (2019) <https://arxiv.org/pdf/1903.07435.pdf>`_;
        * The NPI corpus of `Warstadt et al., (2019) <https://arxiv.org/abs/1909.02597>`_;
        * The stereotypically gendered anaphora resolution corpus of `Jumelet et al., (2019) <https://www.aclweb.org/anthology/K19-1001/>`_, based on the original WinoBias corpus of `Zhao et al., (2018) <https://arxiv.org/abs/1804.06876>`_.

        Furthermore, the current implementation permits similar types of tasks to be easily added, and we plan on incorporating a larger set of tasks in the near future.

    probe
        We provide easy tooling for training diagnostic classifiers (`Hupkes et al., 2018 <https://arxiv.org/pdf/1711.10203.pdf>`_) on top of extracted activations, to probe for linguistic information that might be embedded within them.
        Our extraction module facilitates training diagnostic classifiers on top of intermediate activations as well, including gate activations.
        In recent years it has been pointed out that a high probing accuracy does not necessarily imply that linguistic information is actively being encoded by a model.
        To address this we have incorporated functionality for Control Tasks (`Hewitt & Liang, 2019 <https://arxiv.org/pdf/1909.03368.pdf>`_), providing more qualitative insights.

    attribute
        We provide functionality for model-agnostic feature attributions, that allow the output of a model to be decomposed into a sum of contributions.
        This is achieved by implementing a wrapper over the operations of PyTorch, allowing intermediate feature contributions to be propagated during a forward pass in the model.
        Our implementation provides a basis for many Shapley-based attribution methods, as it allows different approximation methods to be tested easily.
        We currently facilitate the approximation procedure of (Generalised) Contextual Decomposition (`Murdoch et al., 2018 <https://arxiv.org/pdf/1801.05453.pdf>`_), as well as the exact computation of propagated Shapley values.
        Our implementation is the first model-agnostic implementation of Contextual Decomposition: previous implementations were dependent on a fixed model structure.


.. toctree::
   :maxdepth: 2
   :caption: Documentation overview:

   diagnnose.activations
   diagnnose.attribute
   diagnnose.config
   diagnnose.corpus
   diagnnose.extract
   diagnnose.models
   diagnnose.models.wrappers
   diagnnose.probe
   diagnnose.syntax
   diagnnose.syntax.tasks
   diagnnose.tokenizer
   diagnnose.utils
   custom_typedefs
