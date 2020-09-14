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

task_constructors: Dict[str, Type[DownstreamTask]] = {
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
    config : Dict[str, Any]
        Dictionary mapping a task name (`lakretz`, `linzen`, `marvin`,
        or `winobias`) to its configuration (`path`, `tasks`, and
        `task_activations`). `path` points to the corpus folder of the
        task, `tasks` is an optional list of subtasks, and
        `task_activations` an optional path to the folder containing
        the model activations.
    tokenizer : PreTrainedTokenizer
        Tokenizer that converts tokens to their index within the LM.
    """

    def __init__(
        self,
        model: LanguageModel,
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
        ignore_unk: bool = False,
        tasks: Optional[List[str]] = None,
    ) -> None:
        self.ignore_unk = ignore_unk
        self.tasks: Dict[str, DownstreamTask] = {}

        print("Initializing downstream tasks...")

        for task_name in tasks or config.keys():
            task_config = config[task_name]

            # If a single subtask is passed as cmd arg it is not converted to a list yet
            subtasks = task_config.pop("subtasks", None)
            if isinstance(subtasks, str):
                subtasks = [subtasks]

            constructor = task_constructors[task_name]
            self.tasks[task_name] = constructor(
                model, tokenizer, task_config.pop("path"), subtasks=subtasks
            )

        print("Downstream task initialization finished")

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        results: Dict[str, ResultsDict] = {}

        for task_name, task in self.tasks.items():
            print(f"\n--=={task_name.upper()}==--")

            results[task_name] = task.run(ignore_unk=self.ignore_unk, **kwargs)

        return results
