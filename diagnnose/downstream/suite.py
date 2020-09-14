from typing import Any, Dict, List, Optional, Type

from transformers import PreTrainedTokenizer

from diagnnose.downstream.tasks import (
    DownstreamTask,
    LakretzDownstream,
    LinzenDownstream,
    MarvinDownstream,
    ResultsDict,
    WarstadtDownstream,
    WinobiasDownstream,
)
from diagnnose.models import LanguageModel

task_constructors: Dict[str, Callable] = {
    "lakretz": LakretzDownstream,
    "linzen": LinzenDownstream,
    "marvin": MarvinDownstream,
    "warstadt": WarstadtDownstream,
    "winobias": WinobiasDownstream,
}


class DownstreamSuite:
    """Suite that runs multiple downstream tasks on a LM.

    Tasks can be run on already extracted activations, or on a new LM
    for which new activations will be extracted.

    Initialisation is performed separately from the tasks themselves,
    in order to allow multiple LMs to be ran on the same set of tasks.

    Parameters
    ----------
    downstream_config : Dict[str, Any]
        Dictionary mapping a task name (`lakretz`, `linzen`, `marvin`,
        or `winobias`) to its configuration (`path`, `tasks`, and
        `task_activations`). `path` points to the corpus folder of the
        task, `tasks` is an optional list of subtasks, and
        `task_activations` an optional path to the folder containing
        the model activations.
    tokenizer : Tokenizer
        Tokenizer that converts tokens to their index within the LM.
    decompose_config : Dict[str, Any], optional
        Optional setup to perform contextual decomposition on the
        activations prior to executing the downstream tasks.
    """

    def __init__(
        self,
        model: LanguageModel,
        downstream_config: Dict[str, Any],
        tokenizer: Tokenizer,
        decompose_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.downstream_config = downstream_config
        self.decompose_config = decompose_config

        self.tasks: Dict[str, DownstreamTask] = {}

        print("Initializing downstream tasks...")

        for task_name, config in self.downstream_config.items():
            # If a single subtask is passed as cmd arg it is not converted to a list yet
            subtasks = config.pop("subtasks", None)
            if isinstance(subtasks, str):
                subtasks = [subtasks]

            constructor = task_constructors[task_name]
            self.tasks[task_name] = constructor(
                model, tokenizer, config.pop("path"), subtasks=subtasks
            )

        print("Downstream task initialization finished")

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        results: Dict[str, ResultsDict] = {}

        for task_name, task in self.tasks.items():
            print(f"\n--=={task_name.upper()}==--")

            results[task_name] = task.run(**kwargs)

        return results
