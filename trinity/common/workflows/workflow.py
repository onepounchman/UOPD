# -*- coding: utf-8 -*-
"""Base Workflow Class"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, List, Optional, Type, Union

from trinity.common.config import FormatConfig, GenerationConfig
from trinity.common.experience import Experience
from trinity.common.rewards.reward_fn import RewardFn
from trinity.utils.log import get_logger

if TYPE_CHECKING:
    import openai

    from trinity.common.models.model import ModelWrapper


@dataclass
class Task(dict):
    """A Task class that defines a task and its associated reward function / workflow."""

    workflow: Type[Workflow] = None
    repeat_times: Optional[int] = None
    format_args: FormatConfig = field(default_factory=FormatConfig)
    rollout_args: GenerationConfig = field(default_factory=GenerationConfig)
    workflow_args: dict = field(default_factory=dict)
    reward_fn_args: dict = field(default_factory=dict)
    is_eval: bool = False
    reward_fn: Optional[Type[RewardFn]] = None
    raw_task: Optional[dict] = None  # The raw data sample

    # automatically assigned ids
    batch_id: Union[int, str] = ""
    task_id: Union[int, str] = ""

    index: dict = field(default_factory=dict)

    def to_workflow(
        self,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ) -> Workflow:
        """Convert the task to a workflow.

        Args:
            model (ModelWrapper): The rollout model for the workflow.
            auxiliary_models (List[ModelWrapper]): The auxiliary model wrappers.
                Workflows can access both the ModelWrapper and OpenAI client via
                self.auxiliary_model_wrappers and self.auxiliary_models respectively.

        Returns:
            Workflow: The generated workflow object.
        """
        return self.workflow(
            model=model,
            task=self,
            auxiliary_models=auxiliary_models,
        )

    # Deprecated property, will be removed in the future
    @property
    def task_desc(self) -> Union[str, None]:
        prompt_key = self.format_args.prompt_key
        return self.raw_task[prompt_key] if prompt_key in self.raw_task else None  # type: ignore

    # Deprecated property, will be removed in the future
    @property
    def truth(self) -> Union[str, None]:
        response_key = self.format_args.response_key
        return self.raw_task[response_key] if response_key in self.raw_task else None  # type: ignore

    def to_dict(self) -> dict:
        return self.raw_task  # type: ignore


class Workflow:
    """The base workflow class.

    A workflow is a runnable object which generates a list of experiences.

    Attributes:
        auxiliary_model_wrappers: List of ModelWrapper instances for auxiliary models.
        auxiliary_models: List of OpenAI clients (sync or async based on is_async) for auxiliary models.
    """

    can_reset: bool = False  # whether the workflow can be reset with a new task. If true, `reset()` must be implemented.
    can_repeat: bool = False  # whether the workflow can be repeated multiple times. If true, `set_repeat_times()` must be implemented.
    is_async: bool = False  # whether the workflow runs in async mode. If true, `run_async()` must be implemented, else `run()` must be implemented.

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        self.task = task
        self.model = model
        # Store ModelWrapper instances
        self.auxiliary_model_wrappers = auxiliary_models
        # Get OpenAI clients from ModelWrapper (async or sync based on workflow type)
        self.auxiliary_models: Optional[Union[List[openai.OpenAI], List[openai.AsyncOpenAI]]] = None
        if auxiliary_models:
            if self.__class__.is_async:
                self.auxiliary_models = [m.get_openai_async_client() for m in auxiliary_models]
            else:
                self.auxiliary_models = [m.get_openai_client() for m in auxiliary_models]
        self.run_id_base = 0
        self.logger = get_logger(__name__)

    @property
    def resettable(self):
        """Deprecated, use cls.can_reset instead."""
        return self.__class__.can_reset

    @property
    def repeatable(self):
        """Deprecated, use cls.can_repeat instead.
        A workflow is repeatable if it can be run multiple times within the run() or run_async() method.
        """
        return self.__class__.can_repeat

    @property
    def asynchronous(self):
        """Deprecated, use cls.is_async instead.
        Whether the workflow runs in async mode."""
        return self.__class__.is_async

    def reset(self, task: Task):
        """Reset the workflow."""
        raise NotImplementedError

    def set_repeat_times(self, repeat_times: int, run_id_base: int) -> None:
        """
        Set the number of times to repeat the workflow.
        Args:
            repeat_times (int): number of times to repeat the workflow (if repeatable).
            run_id_base (int): base run_id for setting run_id in experiences.
        """
        raise NotImplementedError(
            "set_repeat_times() must be implemented for a repeatable workflow."
        )

    def run(self) -> List[Experience]:
        """Run workflow and return a list of experiences."""
        raise NotImplementedError

    async def run_async(self) -> List[Experience]:
        """Run workflow in async and return a list of experiences."""
        raise NotImplementedError
