from functools import reduce
from typing import Iterable, List

from torchtext.data import Example
from transformers import PreTrainedTokenizer

from diagnnose.typedefs.activations import SelectionFunc


def return_all(_w_idx: int, _item: Example) -> bool:
    """ Always returns True for every token. """
    return True


def final_token(sen_column: str = "sen") -> SelectionFunc:
    """Only returns the final token of a sentence.

    Wrapper allows a different ``sen_column`` to be set, that indicates
    the ``sen`` attribute of a corpus item that is being processed.
    """

    def selection_func(w_idx: int, item: Example) -> bool:
        sen = getattr(item, sen_column)

        return w_idx == (len(sen) - 1)

    return selection_func


def final_sen_token(w_idx: int, item: Example) -> bool:
    """ Only returns the final token of a sentence. """
    sen = getattr(item, "sen")

    return w_idx == (len(sen) - 1)


def only_mask_token(mask_token: str, sen_column: str = "sen") -> SelectionFunc:
    def selection_func(w_idx: int, item: Example) -> bool:
        sen = getattr(item, sen_column)

        return sen[w_idx] == mask_token

    return selection_func


def no_special_tokens(
    tokenizer: PreTrainedTokenizer, sen_column: str = "sen"
) -> SelectionFunc:
    def selection_func(w_idx: int, item: Example) -> bool:
        sen = getattr(item, sen_column)

        try:
            return sen[w_idx] not in tokenizer.all_special_tokens
        except IndexError:
            raise

    return selection_func


def first_n(n: int) -> SelectionFunc:
    """Wrapper that creates a selection_func that only returns True for
    the first `n` items of a corpus.
    """

    def selection_func(_w_idx: int, item: Example) -> bool:
        return item.sen_idx < n

    return selection_func


def nth_token(n: int) -> SelectionFunc:
    """Wrapper that creates a selection_func that only returns True for
    the `n^{th}` token of a sentence.
    """

    def selection_func(w_idx: int, _item: Example) -> bool:
        return w_idx == n

    return selection_func


def in_sen_ids(sen_ids: List[int]) -> SelectionFunc:
    """Wrapper that creates a selection_func that only returns True for
    a `sen_id` if it is part of the provided list of `sen_ids`.
    """

    def selection_func(_w_idx: int, item: Example) -> bool:
        return item.sen_idx in sen_ids

    return selection_func


# Higher-order boolean selection_func logic
def intersection(selection_funcs: Iterable[SelectionFunc]) -> SelectionFunc:
    """ Returns the intersection of an iterable of selection_funcs. """

    def selection_func(w_idx: int, item: Example) -> bool:
        return reduce(
            lambda out, func: out and func(w_idx, item), selection_funcs, True
        )

    return selection_func


def union(selection_funcs: Iterable[SelectionFunc]) -> SelectionFunc:
    """ Returns the union of an iterable of selection_funcs. """

    def selection_func(w_idx: int, item: Example) -> bool:
        return reduce(
            lambda out, func: out or func(w_idx, item), selection_funcs, False
        )

    return selection_func


def negate(selection_func: SelectionFunc) -> SelectionFunc:
    """ Returns the negation of a selection_func. """

    def neg_selection_func(w_idx: int, item: Example) -> bool:
        return not selection_func(w_idx, item)

    return neg_selection_func
