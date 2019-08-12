from typing import Any, Callable, Dict, List, Optional

from diagnnose.downstream.lakretz import lakretz_downstream
from diagnnose.downstream.marvin import marvin_downstream
from diagnnose.downstream.linzen import linzen_downstream
from diagnnose.models.lm import LanguageModel


task_defs: Dict[str, Callable] = {
    "lakretz": lakretz_downstream,
    "marvin": marvin_downstream,
    "linzen": linzen_downstream,
}


# TODO: add docstring
class DownstreamSuite:
    def __init__(
        self,
        downstream_config: Dict[str, Any],
        device: str = "cpu",
        print_results: bool = True,
    ) -> None:
        self.downstream_config = downstream_config
        self.device = device
        self.print_results = print_results

    def perform_tasks(self, model: LanguageModel, vocab_path: str) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        model.eval()

        for task, config in self.downstream_config.items():
            if self.print_results:
                print(f"\n--=={task.upper()}==--")

            results[task] = task_defs[task](
                model,
                vocab_path,
                config["path"],
                tasks=config.get("tasks", None),
                task_activations=config.get("task_activations", None),
                device=self.device,
                print_results=self.print_results,
            )

        return results
