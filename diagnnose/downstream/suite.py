from typing import Any, Callable, Dict, Optional

from diagnnose.downstream.lakretz import lakretz_downstream, lakretz_init
from diagnnose.downstream.linzen import linzen_downstream, linzen_init
from diagnnose.downstream.marvin import marvin_downstream, marvin_init
from diagnnose.downstream.warstadt.downstream import warstadt_downstream, warstadt_init
from diagnnose.downstream.winobias import winobias_downstream, winobias_init
from diagnnose.typedefs.models import LanguageModel
from diagnnose.utils.misc import suppress_print

task_inits: Dict[str, Callable] = {
    "lakretz": lakretz_init,
    "linzen": linzen_init,
    "marvin": marvin_init,
    "warstadt": warstadt_init,
    "winobias": winobias_init,
}

task_defs: Dict[str, Callable] = {
    "lakretz": lakretz_downstream,
    "marvin": marvin_downstream,
    "linzen": linzen_downstream,
    "warstadt": warstadt_downstream,
    "winobias": winobias_downstream,
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
    device : str, optional
        Torch device on which forward passes will be run.
        Defaults to cpu.
    print_results : bool, optional
        Toggle to print task results. Defaults to True.
    """

    def __init__(
        self,
        downstream_config: Dict[str, Any],
        vocab_path: str,
        decompose_config: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        print_results: bool = True,
    ) -> None:
        self.downstream_config = downstream_config
        self.decompose_config = decompose_config
        self.print_results = print_results

        self.init_dicts: Dict[str, Any] = {}
        for task, config in self.downstream_config.items():
            # If a single subtask is passed as cmd arg it is not converted to a list yet
            subtasks = config.pop("subtasks", None)
            if isinstance(subtasks, str):
                subtasks = [subtasks]

            self.init_dicts[task] = task_inits[task](
                vocab_path,
                config.pop("path"),
                subtasks=subtasks,
                task_activations=config.pop("task_activations", None),
                device=device,
                **config,
            )

    def run(self, model: LanguageModel, **kwargs: Any) -> Dict[str, Any]:
        if not self.print_results:
            return self.perform_tasks_wo_print(model, **kwargs)
        return self.perform_tasks(model, **kwargs)

    def perform_tasks(self, model: LanguageModel, **kwargs: Any) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        model.eval()

        for task, config in self.downstream_config.items():
            print(f"\n--=={task.upper()}==--")

            results[task] = task_defs[task](
                self.init_dicts[task],
                model,
                decompose_config=self.decompose_config,
                **config,
                **kwargs,
            )

        return results

    @suppress_print
    def perform_tasks_wo_print(
        self, model: LanguageModel, **kwargs: Any
    ) -> Dict[str, Any]:
        return self.perform_tasks(model, **kwargs)
