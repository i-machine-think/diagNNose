from typing import Dict, Union

import pandas as pd

from diagnnose.corpus import Corpus

# subtask -> Corpus | (condition -> Corpus)
SyntaxEvalCorpora = Dict[str, Union[Corpus, Dict[str, Corpus]]]

# subtask -> accuracy | (condition -> accuracy)
AccuracyDict = Dict[str, Union[float, Dict[str, float]]]
ScoresDict = Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]
