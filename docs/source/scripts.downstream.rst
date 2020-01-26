Downstream tasks
================

This script demonstrates the minimal setup for running the linguistic downstream tasks.

It can be executed using ``python3 downstream.py -c downstream.json``.


Configuration
^^^^^^^^^^^^^
We first consider the expected setup of the ``downstream.json`` config file:

.. literalinclude:: ../../scripts/downstream.json
    :language: json
    :emphasize-lines: 29-32

It can be seen that the file is split up in several sections:

* ``model`` contains the model ``type``, that should be set to one of the model types in `model_wrappers <diagnnose.model_wrappers.html>`_.
  Note that ``model.state_dict`` should point to the pickled ``state_dict`` of the model, and not the pickled class instance.
* ``init_states.corpus_path`` is an optional argument that points to a corpus for which the end-of-sequence hidden states will be used at initialisation.
* ``vocab.path`` points to a ``.txt`` file that contains a vocab entry at each line.
* ``downstream`` contains the arguments that are passed to the `DownstreamSuite <diagnnose.downstream.html#diagnnose.downstream.suite.DownstreamSuite>`_ class.
  It consists of a list of ``tasks``, that tell the suite which tasks should be executed.
  Note that this list is optional, if it is not provided the keys of ``downstream.config`` will be used instead.

  The configuration of each task is provided in ``config``, that maps each task to its configuration.
  Each task configuration should at least contain a ``path``, that points to the location of the directory or file of this specific task.
  If a task consists of several ``subtasks`` these can optionally be provided as a list, as is highlighted for the ``lakretz`` task.

  Optionally the task activations can be passed, in order to save time.
  ``task_activations`` should map a subtask to the directory containing its activations, that have been extracted with the ``Extractor``.

The following papers/corpora are currently implemented:

* `Linzen et al. (2016) <https://arxiv.org/pdf/1611.01368.pdf>`_ -- The landmark paper that instigated this field of research.
  Currently expects corpus file to be of a specific `.tsv` format which can be downloaded `here <https://www.dropbox.com/s/ajvj9ackil18ps0/1attr_sample.tsv?dl=0>`_.
  Full corpus support will be added soon.
* `Marvin & Linzen (2018) <https://arxiv.org/pdf/1808.09031.pdf>`_ -- Broad array of linguistic phenomena, all of which are supported.
  Data can be found in `their repository <https://github.com/BeckyMarvin/LM_syneval/tree/master/data/templates>`_, and the ``downstream.config.marvin.path`` field in ``downstream.json`` is expected to point to a directory containing these ``.pickle`` files.
* `Lakretz et al. (2019) <https://www.aclweb.org/anthology/N19-1002.pdf>`_ -- Extensive investigation into subject-verb agreement, based on a useful set of artifical constructions.
  Data can be found in `this repository <https://github.com/FAIRNS/Number_and_syntax_units_in_LSTM_LMs/tree/master/datasets/NA_tasks>`_, and the ``downstream.config.lakretz.path`` field in ``downstream.json`` is expected to point to a directory containing these ``.txt`` files.
* `Warstadt et al. (2019) <https://arxiv.org/pdf/1909.02597.pdf>`_ -- Extensive investigation into BERTs capacities of handling Negative Polarity Items, based on a richly constructed corpus.
  Data can be found `in this .zip <https://alexwarstadt.files.wordpress.com/2019/08/npi_lincensing_data.zip>`_, which contains a ``.tsv`` towards which ``downstream.config.warstadt.path`` in ``downstream.json`` is expected to point.


Script
^^^^^^
The `downstream.py <https://github.com/i-machine-think/diagnnose/blob/master/scripts/downstream.py>`_ script is structured as follows.

.. literalinclude:: ../../scripts/downstream.py
    :lines: 10-16
    :lineno-start: 10
    :linenos:

We first create the configuration setup.
``arg_groups`` contains the section that should be provided in the ``.json`` config file.
Based on these groups an ``arg_parser`` is initialised that creates a ``config_dict``.
This dictionary maps each group to the provided arguments.

.. literalinclude:: ../../scripts/downstream.py
    :lines: 26-31
    :lineno-start: 26
    :linenos:

We create a `DownstreamSuite <diagnnose.downstream.html#diagnnose.downstream.suite.DownstreamSuite>`_ based on the provided configuration.
Note that the full ``config_dict`` is passed, as the initialisation depends on multiple sections of the ``config_dict``.
We then extract the ``vocab_path`` and pass this to `import_corpus <diagnnose.corpus.html#diagnnose.corpus.import_corpus.import_corpus>`_.

.. literalinclude:: ../../scripts/downstream.py
    :lines: 33-34
    :lineno-start: 33
    :linenos:

To run the tasks in a suite we need to provide a ``LanguageModel`` instance.
This allows a suite to be ran on different LMs.
Setting ``ignore_unk`` allows sentences to be skipped if they contain tokens that are not part of the model vocabulary.

The ``results`` that are returned using ``suite.run`` are a dictionary that maps each task to the model accuracy.
