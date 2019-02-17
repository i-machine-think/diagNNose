from typing import Callable

from .corpus import LabeledSentence

SelectFunc = Callable[[int, str, LabeledSentence], bool]
