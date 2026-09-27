# -*- coding: utf-8 -*-
"""Teacher-free evaluation workflow for WebShop (TCOD paper).

Reuses the EXACT rollout loop / prompt templates / action parsing / reward of
``OPD_webshop_workflow`` (so eval numbers are directly comparable to the OPD/TCOD
training setup), but:

  * does NOT require a teacher / auxiliary model (no distillation here), and
  * skips the teacher-logprob / KL computation entirely.

Per episode it records the WebShop metrics:

  * ``env_done``   -> 1.0 if the episode terminated (agent bought), else 0.0
  * ``env_rounds`` -> number of interaction rounds
  * ``score``      -> WebShop matching score in [0, 1] (final env reward)
  * ``success``    -> 1.0 if score >= 1.0 (perfect match), else 0.0

The explorer aggregates these across an eval taskset as
``bench/<taskset>/score/mean@1`` etc. Use via
``default_workflow_type: 'eval_webshop_workflow'`` in a ``mode: bench`` config.
"""

from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task, Workflow
from trinity.common.workflows.envs.UOPD.webshop.OPD_workflow import (
    OnPolicyDistillVerlAgentWebshopWorkflow,
)
from trinity.common.workflows.envs.UOPD.webshop.utils import (
    HISTORY_LENGTH,
    WEBSHOP_TEMPLATE,
    WEBSHOP_TEMPLATE_NO_HIS,
    _create_webshop_env,
    _extract_task_description,
    _format_available_actions,
    _format_history,
    format_observation,
    parse_action,
    validate_action,
)


@WORKFLOWS.register_module("eval_webshop_workflow")
class EvalWebshopWorkflow(OnPolicyDistillVerlAgentWebshopWorkflow):
    """Single-model WebShop rollout for evaluation (no teacher required)."""

    is_async: bool = True
    can_reset: bool = True
    can_repeat: bool = False

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        # Skip OPD.__init__ (which asserts a teacher); call the base Workflow init.
        Workflow.__init__(
            self,
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.reset(task)
        self.teacher_model = None  # no teacher in eval
        self.temperature = task.workflow_args.get("temperature", 1.0)
        self.max_env_steps = task.workflow_args.get("max_env_steps", 15)
        self.env = _create_webshop_env()

    async def run_async(self) -> List[Experience]:
        session_id = int(self.task_desc)
        return await self._run_episode(
            session_id=session_id, run_id=getattr(self, "run_id_base", 0)
        )

    async def _run_episode(self, session_id: int, run_id: int) -> List[Experience]:
        self.env.reset(session=session_id)
        observation = self.env.observation
        self._env_done = False
        self._env_rounds = 0
        self._final_reward = 0.0

        task_description = _extract_task_description(observation)
        history: List[str] = []
        memory = self.format_messages()
        turn_responses: List[Experience] = []

        kwargs = {**self.rollout_args, "n": 1}

        for r in range(self.max_env_steps):
            available_actions = self.env.get_available_actions()
            formatted_observation = format_observation(observation)
            formatted_actions = _format_available_actions(available_actions)

            if len(history) < HISTORY_LENGTH:
                user_content = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=task_description,
                    current_observation=formatted_observation,
                    available_actions=formatted_actions,
                )
            else:
                action_history_str = "\n".join(history[-HISTORY_LENGTH:])
                user_content = WEBSHOP_TEMPLATE.format(
                    task_description=task_description,
                    step_count=r,
                    history_length=min(HISTORY_LENGTH, len(history)),
                    action_history=action_history_str,
                    current_step=r + 1,
                    current_observation=formatted_observation,
                    available_actions=formatted_actions,
                )

            memory = [{"role": "user", "content": user_content}]

            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            turn_responses.append(response)

            action = parse_action(response_text)
            action_valid, error_msg = validate_action(action, available_actions)
            history.append(_format_history(formatted_observation, r + 1, action))

            if action_valid:
                observation, reward, done, _ = self.env.step(action)
            else:
                observation = error_msg
                reward = 0.0
                done = False

            if done:
                self._env_done = True
                self._env_rounds = r + 1
                self._final_reward = float(reward)
                break
        else:
            self._env_rounds = self.max_env_steps

        # Finalize experiences (mirror OPD, minus teacher-logprob / KL).
        for i, response in enumerate(turn_responses):
            if response.metrics is None:
                response.metrics = {}
            response.reward = self.compute_reward(response)
            response.eid.run = run_id
            response.eid.step = i

        if turn_responses:
            last_response = turn_responses[-1]
            if last_response.metrics is None:
                last_response.metrics = {}
            last_response.metrics["env_rounds"] = self._env_rounds
            last_response.metrics["env_done"] = 1.0 if self._env_done else 0.0
            last_response.metrics["score"] = float(self._final_reward)
            last_response.metrics["success"] = 1.0 if self._final_reward >= 1.0 else 0.0
            last_response.metrics["session_id"] = float(session_id)

        # Cross-process progress signal: one line per finished episode appended to
        # $WS_EVAL_PROGRESS. A monitor counts completed episodes to
        # draw a progress bar. Robust across the 16 Ray WorkflowRunner processes.
        import os as _os

        _pf = _os.environ.get("WS_EVAL_PROGRESS")
        if _pf:
            try:
                with open(_pf, "a") as _f:
                    _f.write(
                        f"{session_id}\t{float(self._final_reward):.4f}\t"
                        f"{self._env_rounds}\t{1 if self._env_done else 0}\n"
                    )
            except Exception:
                pass

        return turn_responses
