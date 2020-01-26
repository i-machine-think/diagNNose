Activation extraction
=====================

The following script demonstrates the minimal setup for activation extraction.

It can be executed using ``python3 extract.py -c extract.json``.

Configuration
^^^^^^^^^^^^^
We first consider the expected setup of the ``extract.json`` config file:

.. literalinclude:: ../../scripts/extract.json
    :language: json

It can be seen that the file is split up in several sections:

* ``model`` contains the model ``type``, that should be set to one of the model types in `model_wrappers <diagnnose.model_wrappers.html>`_.
  Note that ``model.state_dict`` should point to the pickled ``state_dict`` of the model, and not the pickled class instance.
* ``init_states.corpus_path`` is an optional argument that points to a corpus for which the end-of-sequence hidden states will be used at initialisation.
* ``vocab.path`` points to a ``.txt`` file that contains a vocab entry at each line.
* ``corpus`` contains the parameters that are passed to the `import_corpus <diagnnose.corpus.html#diagnnose.corpus.import_corpus.import_corpus>`_ method.
* ``activations.activation_names`` is a list containing the activations that will be extracted.
  Items are of the form ``{name}{layer}``.
  Layer counting starts from 0.
  The setup provided here (``"hx1"``) will extract only the activations of the hidden state (``hx``) of the second LSTM layer (``1``).
* ``extract`` contains the arguments that are passed to the `Extractor.extract <diagnnose.extractors.html#diagnnose.extractors.base_extractor.Extractor.extract>`_ method.

Script
^^^^^^
The `extract.py <https://github.com/i-machine-think/diagnnose/blob/master/scripts/extract.py>`_ script is structured as follows.

.. literalinclude:: ../../scripts/extract.py
    :lines: 12-15
    :lineno-start: 12
    :linenos:

We first create the configuration setup.
``arg_groups`` contains the section that should be provided in the ``.json`` config file.
Based on these groups an ``arg_parser`` is initialised that creates a ``config_dict``.
This dictionary maps each group to the provided arguments.

.. literalinclude:: ../../scripts/extract.py
    :lines: 17-19
    :lineno-start: 17
    :linenos:

We create a ``LanguageModel`` by passing the ``config_dict`` to `import_model <diagnnose.models.html#diagnnose.models.import_model.import_model>`_.
Note that the full ``config_dict`` is passed, as the initialisation depends on multiple sections of the ``config_dict``.
We then extract the ``vocab_path`` and pass this to `import_corpus <diagnnose.corpus.html#diagnnose.corpus.import_corpus.import_corpus>`_.

.. literalinclude:: ../../scripts/extract.py
    :lines: 22-23
    :lineno-start: 22
    :linenos:

We can optionally define a ``selection_func``, that allows us to define which exact activations we want to extract.
A ``selection_func`` receives a sentence index, the current position in that sentence, and the ``torchtext.Example`` item that contains additional information.
The ``selection_func`` presented here will only select the last activations of a sentence.

.. literalinclude:: ../../scripts/extract.py
    :lines: 25-26
    :lineno-start: 25
    :linenos:

The ``model`` and ``corpus`` are passed to the ``Extractor`` class, alongside the ``activations`` configuration.
Extraction is performed by calling ``extractor.extract``.

The extracted activations will be stored as pickled tensors in ``activations_activations_dir``.
These can later be accessed using an `ActivationReader <diagnnose.activations.html#diagnnose.activations.activation_reader.ActivationReader>`_.
