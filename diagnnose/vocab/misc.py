from typing import Any, Dict, Optional


def get_vocab_from_config(config_dict: Dict[str, Any]) -> Optional[str]:
    """ Vocab is handled slightly different for GoogleLM. """
    vocab_path: Optional[str] = (
        config_dict["model"].get("corpus_vocab_path", None)
        or config_dict["model"].get("full_vocab_path", None)
        or config_dict["vocab"].get("path", None)
    )

    return vocab_path
