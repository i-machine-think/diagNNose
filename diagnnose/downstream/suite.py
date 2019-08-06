from typing import Any, Dict, List, Optional

from diagnnose.downstream.lakretz import lakretz_downstream
from diagnnose.downstream.marvin import marvin_downstream
from diagnnose.models.lm import LanguageModel


# TODO: add docstring
class DownstreamSuite:
    def __init__(
        self,
        task_activations: Optional[Dict[str, Dict[str, str]]] = None,
        lakretz_path: Optional[str] = None,
        lakretz_tasks: Optional[List[str]] = None,
        marvin_path: Optional[str] = None,
        marvin_tasks: Optional[List[str]] = None,
        device: str = "cpu",
        print_results: bool = True,
    ) -> None:
        self.task_activations = task_activations or {}
        self.lakretz_path = lakretz_path
        self.lakretz_tasks = lakretz_tasks
        self.marvin_path = marvin_path
        self.marvin_tasks = marvin_tasks
        self.device = device
        self.print_results = print_results

    def perform_tasks(self, model: LanguageModel, vocab_path: str) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        model.eval()

        if self.lakretz_path:
            if self.print_results:
                print("\n--==LAKRETZ==--")

            results["lakretz"] = lakretz_downstream(
                model,
                vocab_path,
                self.lakretz_path,
                lakretz_activations=self.task_activations.get("lakretz", None),
                lakretz_tasks=self.lakretz_tasks,
                device=self.device,
                print_results=self.print_results,
            )
        if self.marvin_path:
            if self.print_results:
                print("\n--==MARVIN==--")

            results["marvin"] = marvin_downstream(
                model,
                vocab_path,
                self.marvin_path,
                marvin_activations=self.task_activations.get("marvin", None),
                marvin_tasks=self.marvin_tasks,
                device=self.device,
                print_results=self.print_results,
            )

        return results
