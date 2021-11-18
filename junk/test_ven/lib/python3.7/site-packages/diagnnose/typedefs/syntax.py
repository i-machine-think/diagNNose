from typing import Dict, Union

from diagnnose.corpus import Corpus

# subtask -> Corpus | (condition -> Corpus)
SyntaxEvalCorpora = Dict[str, Union[Corpus, Dict[str, Corpus]]]
# subtask -> accuracy | (condition -> accuracy)
ResultsDict = Dict[str, Union[float, Dict[str, float]]]
