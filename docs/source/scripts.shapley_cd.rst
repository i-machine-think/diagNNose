Shapley Decomposition
=====================

This script demonstrates the minimal setup for running Shapley Decomposition (Master thesis Jaap Jumelet).

It can be executed using ``python3 shapley_cd.py -c shapley_cd.json``.
Note that this script is mainly the same as ``decompose.py``, but describes the procedure using the `ShapleyDecomposer <diagnnose.decompositions.html#module-diagnnose.decompositions.shapley_decomposer>`_.


Configuration
^^^^^^^^^^^^^
We first consider the expected setup of the ``shapley_cd.json`` config file:

.. literalinclude:: ../../scripts/shapley_cd.json
    :language: json

It can be seen that the file is split up in several sections:

* ``model`` contains the model ``type``, that should be set to one of the model types in `model_wrappers <diagnnose.model_wrappers.html>`_.
  Note that ``model.state_dict`` should point to the pickled ``state_dict`` of the model, and not the pickled class instance.
* ``init_states.corpus_path`` is an optional argument that points to a corpus for which the end-of-sequence hidden states will be used at initialisation.
* ``corpus`` contains the parameters that are passed to the `import_corpus <diagnnose.corpus.html#diagnnose.corpus.import_corpus.import_corpus>`_ method.
* ``vocab.path`` points to a ``.txt`` file that contains a vocab entry at each line.
* ``activations.activations_dir`` can point to either: 1) a directory containing extracted activations, or 2) a directory to which new activations will be written if ``decompose.create_new_activations`` is set to ``True``.
* ``decompose`` contains the configuration of the decomposition procedure.
  ``decompose.decomposer`` sets the decomposition type to ``ShapleyDecomposer``.
  These arguments are passed to the `DecomposerFactory <diagnnose.decompositions.html#diagnnose.decompositions.factory.DecomposerFactory>`_.

Script
^^^^^^
The `shapley_cd.py <https://github.com/i-machine-think/diagnnose/blob/master/scripts/shapley_cd.py>`_ script is structured as follows.

.. literalinclude:: ../../scripts/shapley_cd.py
    :lines: 12-15
    :lineno-start: 12
    :linenos:

We first create the configuration setup.
``arg_groups`` contains the section that should be provided in the ``.json`` config file.
Based on these groups an ``arg_parser`` is initialised that creates a ``config_dict``.
This dictionary maps each group to the provided arguments.

.. literalinclude:: ../../scripts/shapley_cd.py
    :lines: 17-19
    :lineno-start: 17
    :linenos:

We create a ``LanguageModel`` by passing the ``config_dict`` to `import_model <diagnnose.models.html#diagnnose.models.import_model.import_model>`_.
Note that the full ``config_dict`` is passed, as the initialisation depends on multiple sections of the ``config_dict``.
We then extract the ``vocab_path`` and pass this to `import_corpus <diagnnose.corpus.html#diagnnose.corpus.import_corpus.import_corpus>`_.

.. literalinclude:: ../../scripts/shapley_cd.py
    :lines: 21-22,25-29
    :lineno-start: 23
    :linenos:

We define the ``sen_ids`` that define the sentences for which the decomposition will be computed.
We then create a `DecomposerFactory <diagnnose.decompositions.html#diagnnose.decompositions.factory.DecomposerFactory>`_ that will create a ``decomposer`` instance.
The ``factory`` also takes care of reading in the required activations, and setting up other properties that are needed for decomposition.

.. literalinclude:: ../../scripts/shapley_cd.py
    :lines: 31-33
    :lineno-start: 31
    :linenos:

The ``factory`` creates a new ``decomposer`` instance, that will compute the decomposition itself.
We again pass along the ``sen_ids``, next to an optional ``classes`` tensor.
This tensor defines the decoder classes for which we will compute the decomposed scores (:math:`\beta^z_T`).
If ``classes`` is not provided the decomposed hidden state will be returned (:math:`\beta^h_T`).

``subsen_index`` defines the token indices for which the decomposition will be created.
For instance, if we only need to create the decomposition up to the 4th step we pass along ``slice(0, 4)``.

.. literalinclude:: ../../scripts/shapley_cd.py
    :lines: 37-40
    :lineno-start: 37
    :linenos:

The actual decomposition is performed by calling ``decompser.decompose``.
The first argument defines the individual partitions.
If this is set to ``None`` the partition will be created for each individual input token.
Otherwise a list can be passed, containing lists (indicating indices that are part of that partition) or tuples (indicating the range that constitutes that partition).

The ``partition`` object that is a tensor that represents the partition.
If ``classes`` has been provided the partitions is a tensor of shape :math:`bsz\times P\times T\times C`, with :math:`bsz` the batch size (number of ``sen_ids``), :math:`P` the number of partitions, :math:`T` the sequence length (length of ``subsen_idx``), and :math:`C` the number of ``classes``.
If ``classes`` has not been provided the partition is a tensor of shape :math:`bsz\times P\times T\times d_h``, with :math:`d_h` the dimension of the hidden state.
