diagnnose documentation
=======================

``diagnnose`` is a Python library containing a set of modules for analysing recurrent neural networks. In particular, it contains functionality for :

* **Extracting activations** from different types of (language) models and providing quick access to these stored activations.
* Running **diagnostic classifiers** (`Hupkes et al., 2018 <https://arxiv.org/pdf/1711.10203.pdf>`_) on extracted activations.
* Running **control tasks** (`Hewitt & Liang, 2019 <https://arxiv.org/pdf/1909.03368.pdf>`_) parallel to these diagnostic classifiers.
* Performing **contextual decomposition** (`Murdoch et al., 2018 <https://arxiv.org/pdf/1801.05453.pdf>`_) on a model.
* Running a broad linguistic `downstream task suite <diagnnose.downstream.html>`_ on a language model.

The library is structured into the following modules:

.. glossary::

    activations
        Provides tools to easily access extracted activations, using an `ActivationReader <diagnnose.activations.html#diagnnose.activations.activation_reader.ActivationReader>`_.

    classifiers
        Provides tools for running diagnostic classifiers, based on extracted activations.
        Also allows `control tasks` to be ran alongside a classifier, to evaluate the performance of the classifier.

    config
        We define a custom argparsing configuration, that allows arguments to be provided in a clear and structured manner.

    corpus
        Functionality for processing a corpus, and tokenizing it according to the model vocabulary. Functionality is based on ``torchtext``.

    decompositions
        We provide implementations for Contextual Decomposition (Murdoch et al, 2018), and propose several improvements of this attribution method.

    downstream
        Allows linguistic downstream tasks to be ran for a language model, to assess its understanding of certain linguistic phenomena such as **subject-verb agreement**, **anaphora resolution**, and **negative polarity items**.

    extraction
        The core of the library that handles the extraction of activations of a model. Extraction is handled by the `Extractor <diagnnose.extractors.html#diagnnose.extractors.base_extractor.Extractor>`_ class, based on a model and a corpus.

    models
        We provide support for various LSTM-based language models, such as the AWD-LSTM (Merity et al., 2017), the GoogleLM (Józefowicz et al., 2016) and the LM of Gulordava et al. (2018).

.. toctree::
   :maxdepth: 2
   :caption: Documentation overview:

   scripts

.. toctree::
   :maxdepth: 2

   diagnnose_
