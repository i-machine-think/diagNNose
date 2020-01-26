diagnnose documentation
=======================

``diagnnose`` is a Python library containing a set of modules for analysing recurrent neural networks. In particular, it contains functionality for :

* **Extracting activations** from different types of (language) models and providing quick access to these stored activations.
* Running **diagnostic classifiers** (`Hupkes et al., 2018 <https://arxiv.org/pdf/1711.10203.pdf>`_) on extracted activations.
* Running **control tasks** (`Hewitt & Liang, 2019 <https://arxiv.org/pdf/1909.03368.pdf>`_) parallel to these diagnostic classifiers.
* Performing **contextual decomposition** (`Murdoch et al., 2018 <https://arxiv.org/pdf/1801.05453.pdf>`_) on a model.
* Running a broad linguistic `downstream task suite <diagnnose.downstream.html>`_ on a language model.

The library is structured as follows:

.. glossary::

   `extraction <diagnnose.extractors.html>`_
        The core of the library that handles the extraction of activations
        of a model. Extraction is handled by the `Extractor <diagnnose.extractors.html#diagnnose.extractors.base_extractor.Extractor>`_
        class, based on a model and a corpus.

   source directory
      The directory which, including its subdirectories, contains all
      source files for one Sphinx project.

.. toctree::
   :maxdepth: 2
   :caption: Library overview:

   scripts

.. toctree::
   :maxdepth: 4

   modules
