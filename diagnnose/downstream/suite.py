from typing import Any, Callable, Dict, Optional

from diagnnose.downstream.lakretz import lakretz_downstream, lakretz_init
from diagnnose.downstream.linzen import linzen_downstream, linzen_init
from diagnnose.downstream.marvin import marvin_downstream, marvin_init
from diagnnose.downstream.winobias import winobias_downstream, winobias_init
from diagnnose.models.lm import LanguageModel
from diagnnose.utils.misc import suppress_print

task_inits: Dict[str, Callable] = {
    "lakretz": lakretz_init,
    "linzen": linzen_init,
    "marvin": marvin_init,
    "winobias": winobias_init,
}

task_defs: Dict[str, Callable] = {
    "lakretz": lakretz_downstream,
    "marvin": marvin_downstream,
    "linzen": linzen_downstream,
    "winobias": winobias_downstream,
}


# TODO: add docstring
class DownstreamSuite:
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
            self.init_dicts[task] = task_inits[task](
                vocab_path,
                config["path"],
                tasks=config.get("tasks", None),
                task_activations=config.get("task_activations", None),
                device=device,
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
                **kwargs,
            )

        return results

    @suppress_print
    def perform_tasks_wo_print(
        self, model: LanguageModel, **kwargs: Any
    ) -> Dict[str, Any]:
        return self.perform_tasks(model, **kwargs)
