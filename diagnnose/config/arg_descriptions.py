from typing import Any, Dict

# Dict mapping: group -> arg_name -> arg_attr -> value
# Where arg_attr one of 'help', 'nargs', or 'type'.
arg_descriptions: Dict[str, Dict[str, Dict[str, Any]]] = {}

arg_descriptions["model"] = {
    # Gulordava/Forward LSTM
    "model_type": {"help": "RNN model type, as of now either ForwardLSTM or GoogleLM."},
    "state_dict": {"help": "Path to model parameters (pickled torch state_dict)"},
    "device": {
        "help": "(optional) Torch device name on which model will be run. Defaults to cpu."
    },
    "rnn_name": {"help": "(optional) Attribute name of rnn, defaults to `rnn`."},
    "encoder_name": {
        "help": "(optional) Attribute name of model encoder, defaults to `encoder`."
    },
    "decoder_name": {
        "help": "(optional) Attribute name of model decoder, defaults to `decoder`."
    },
    # Huggingface Transformer models
    "model_name": {"help": "Huggingface model name of pretrained LM."},
    # GoogleLM
    "pbtxt_path": {
        "help": "Path to the .pbtxt file containing the GraphDef model setup."
    },
    "ckpt_dir": {"help": "Path to folder containing parameter checkpoint files."},
    "full_vocab_path": {
        "help": "Path to the full model vocabulary of 800k tokens. Note that `vocab_path` "
        "can be passed along as well, pointing toward the corpus that will be extracted. "
        "In that case only a subset of the model softmax will be loaded in."
    },
    "corpus_vocab_path": {
        "help": "(optional) Path to the corpus for which a vocabulary will be created. "
        "This allows for only a subset of the model softmax to be loaded in."
    },
    "create_decoder": {
        "help": "(optional) Toggle to load in the (partial) softmax weights. Can be set to "
        "false in case no decoding projection needs to be made, as is the case during "
        "activation extraction, for example."
    },
}

arg_descriptions["tokenizer"] = {
    "path": {
        "help": "Path to the model vocabulary, which should a file containing a vocab "
        "entry at each line."
    },
    "notify_unk": {
        "type": bool,
        "help": "Notify when a requested token is not present in the vocab. Defaults to False.",
        "default": False,
    },
}

arg_descriptions["init_states"] = {
    "pickle_path": {
        "help": "(optional) Path to pickle of the initial lstm states of the model. "
        "If no path is provided zero-initialized states will be used."
    },
    "corpus_path": {
        "help": "(optional) Path to corpus for which the final activation will be used as "
        "initial states to the model."
    },
    "save_init_states_to": {
        "help": "(optional) Path to which the newly computed init_states will be saved. "
        "If not provided these states won't be dumped."
    },
    "use_default": {
        "type": bool,
        "help": "(optional) Toggle to use the default initial sentence `. <eos>`. "
        "Defaults to False",
        "default": False,
    },
}

arg_descriptions["corpus"] = {
    "path": {"help": "(required) Path to a corpus file."},
    "header": {
        "nargs": "*",
        "help": "(optional) List of corpus attribute names. If not provided all lines will be "
        'considered to be sentences, with the attribute name "sen".',
    },
    "to_lower": {
        "type": bool,
        "help": "(optional) Convert corpus to lowercase, defaults to False.",
        "default": False,
    },
    "header_from_first_line": {
        "type": bool,
        "help": "(optional) Use the first line of the corpus as the attribute  names of the "
        "corpus. Defaults to False.",
        "default": False,
    },
    "sep": {
        "help": "Column separator of corpus file, either a tsv or csv. Defaults to '\t'."
    },
}

arg_descriptions["extract"] = {
    "activations_dir": {
        "help": "Path to directory to which extracted embeddings will be written."
    },
    "activation_names": {
        "nargs": "*",
        "help": "List of activation names to be extracted. Activation names must "
        "be provided as a list of lists: i.e. [[layer, name]].",
    },
    "batch_size": {
        "type": int,
        "help": "(optional) Amount of sentences processed per forward step. "
        "Higher batch size increases extraction speed, but should "
        "be done accordingly to the amount of available RAM. Defaults to 1.",
    },
    "dtype": {
        "help": "(optional) Activation dtype, should be one of float32 or float64. "
        "Defaults to float32."
    },
}

arg_descriptions["init_dc"] = {
    "save_dir": {
        "help": "Directory to which trained models will be saved.",
    },
    "test_activations_dir": {
        "help": "(optional) Directory containing the extracted test activations. If not "
        "provided the train activation set will be split and partially used as test set."
    },
    "test_corpus": {
        "help": "(optional) Path to test corpus file, containing the test labels."
    },
    "classifier_type": {
        "help": "Either `logreg_torch`, using a torch logreg model, or  `logreg_sklearn`, "
        "using a LogisticRegressionCV model of sklearn."
    },
    "verbose": {"help": "Set to any positive number for verbosity. Defaults to 0."},
}

arg_descriptions["train_dc"] = {
    "calc_class_weights": {
        "type": bool,
        "help": "(optional) Set to true to calculate the classifier class weights based on the "
        "corpus class frequencies. Defaults to False.",
        "default": False,
    },
    "data_subset_size": {
        "type": int,
        "help": "(optional) Subset size of the amount of data points that will be used for "
        "training. Train/test split is performed afterwards. Defaults to the entire "
        "data set.",
    },
    "train_test_split": {
        "type": float,
        "help": "(optional) Ratio of the train/test split. Defaults to 0.9.",
    },
    "rank": {
        "type": int,
        "help": "Matrix rank of the linear classifier. Defaults to the full rank if not "
        "provided.",
    },
    "max_epochs": {
        "type": int,
        "help": "Maximum number of training epochs used by skorch. Defaults to 10.",
    },
}

arg_descriptions["decompose"] = {
    "decomposer": {
        "help": "Class name of decomposer constructor. As of now either "
        "CellDecomposer or ContextualDecomposer. Defaults to ContextualDecomposer."
    },
    "decompose_o": {
        "help": " Toggles decomposition of the output gate. Defaults to False."
    },
    "rel_interactions": {
        "nargs": "*",
        "help": " Indicates the interactions that are part of the relevant decomposition. "
        "Possible interactions are: rel-rel, rel-b and rel-irrel. Defaults to rel-rel, "
        "rel-irrel & rel-b.",
    },
    "bias_bias_only_in_phrase": {
        "help": " Toggles whether the bias-bias interaction should only be added when inside "
        "the relevant phrase. Defaults to True, indicating that only bias-bias "
        "interactions inside the subphrase range are added to the relevant decomposition."
    },
    "only_source_rel": {
        "help": " Relates to rel-irrel interactions. If set to true, only irrel_gate-rel_"
        "source interactions will be added to rel, similar to LRP (Arras et al., 2017)."
    },
    "only_source_rel_b": {
        "help": " Relates to rel-b interactions. If set to true, only b-rel_source "
        "interactions will be added to rel, similar to LRP (Arras et al., 2017)."
    },
    "input_never_rel": {
        "help": " Never add the Wx input to the rel part, useful when only investigating "
        "the model biases. Defaults to False."
    },
    "init_states_rel": {
        "help": " Directly add the initial cell/hidden states to the relevant part. "
        "Defaults to False."
    },
    "use_extracted_activations": {
        "help": " Allows previously extracted activations to be used to avoid unnecessary "
        "recomputations of those activations. Defaults to True."
    },
    "only_return_dec": {
        "help": " Only returns the decomposed cell states of the top layer, without "
        "calculating the corresponding decoder scores. Defaults to False."
    },
    "normalize": {
        "help": "Toggle to normalize the relevant scores by the full model logits."
    },
    "normalize_w_betas": {
        "help": "Toggle to normalize the relevant scores by the sum of relevant scores."
    },
}

arg_descriptions["plot_attention"] = {
    "title": {"help": "(optional) Title of the plot."},
    "clim": {
        "nargs": "*",
        "help": "(optional) Color limits. Should be be provided as [cmin, cmax], "
        "or [cmin, cmid, cmax].",
    },
    "cmap": {"help": "(optional) Pyplot colormap. Defaults to `coolwarm`."},
    "plot_values": {
        "help": "(optional) Toggle to plot the individual values. Defaults to True."
    },
    "value_font_size": {
        "help": "(optional) Font size of the individual plotted values."
    },
    "patches": {
        "nargs": "*",
        "help": "(optional) List of [x, y] tuples denoting highlighted patch indices.",
    },
}

arg_descriptions["downstream"] = {
    "config": {
        "help": "Dictionary mapping a downstream task to a config dict relating to that task."
        "Current supported tasks are `linzen`, `marvin`, `lakretz`, and `warstadt`."
    },
    "tasks": {
        "nargs": "*",
        "help": "List of tasks for which the downstream setup should be computed. This allows"
        "to store the config of each task in the json if one wishes to skip it momentarily.",
    },
}
