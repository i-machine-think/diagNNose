from typing import Any, Callable, Dict, Optional

from diagnnose.downstream.tasks import (
    LakretzDownstream,
    LinzenDownstream,
    MarvinDownstream,
    WarstadtDownstream,
    WinobiasDownstream,
)
from diagnnose.typedefs.models import LanguageModel
from diagnnose.utils.misc import suppress_print

from .task import DownstreamTask, ResultsDict

task_constructors: Dict[str, Callable] = {
    "lakretz": LakretzDownstream,
    "linzen": LinzenDownstream,
    "marvin": MarvinDownstream,
    "warstadt": WarstadtDownstream,
    "winobias": WinobiasDownstream,
}


class DownstreamSuite:
    """ Suite that runs multiple downstream tasks on a LM.

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
    vocab_path : str
        Path to the vocabulary of the LM. Needs to be provided in order
        to check if a token in a task is part of the model vocabulary.
    decompose_config : Dict[str, Any], optional
        Optional setup to perform contextual decomposition on the
        activations prior to executing the downstream tasks.
    """

    def __init__(
        self,
        model: LanguageModel,
        downstream_config: Dict[str, Any],
        vocab_path: str,
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
                model, vocab_path, config.pop("path"), subtasks=subtasks
            )

        print("Downstream task initialization finished")

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        results: Dict[str, ResultsDict] = {}

        for task_name, task in self.tasks.items():
            print(f"\n--=={task_name.upper()}==--")

            results[task_name] = task.run(**kwargs)

        return results
