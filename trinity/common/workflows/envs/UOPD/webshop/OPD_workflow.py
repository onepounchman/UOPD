# -*- coding: utf-8 -*-
"""On-Policy Distillation (OPD) workflow for WebShop."""

from dataclasses import asdict
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task, Workflow

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

@WORKFLOWS.register_module("OPD_webshop_workflow")
class OnPolicyDistillVerlAgentWebshopWorkflow(Workflow):
    """On-policy distillation workflow for WebShop."""

    is_async: bool = True
    can_reset: bool = True
    can_repeat: bool = True

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.reset(task)

        assert (
            self.auxiliary_model_wrappers is not None
            and len(self.auxiliary_model_wrappers) >= 1
        ), "On-policy distillation requires at least one auxiliary model as teacher."
        self.teacher_model = self.auxiliary_model_wrappers[0]

        self.temperature = task.workflow_args.get("temperature", 1.0)
        self.max_env_steps = task.workflow_args.get("max_env_steps", 15)
        self.env = _create_webshop_env()

    def reset(self, task: Task):
        self.task = task
        self.format_args = task.format_args
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc or "0"
        self.is_eval = task.is_eval
        self.repeat_times = task.repeat_times or 1

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base

    def compute_reward(self, response: Experience) -> float:
        return getattr(self, "_final_reward", 0.0)

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def format_messages(self):
        return []

    async def run_async(self) -> List[Experience]:
        session_id = int(self.task_desc)
        all_turn_responses: List[Experience] = []

        for rollout_idx in range(self.repeat_times):
            rollout_responses = await self._run_episode(
                session_id=session_id,
                run_id=self.run_id_base + rollout_idx,
            )
            all_turn_responses.extend(rollout_responses)

        return all_turn_responses

    async def _run_episode(self, session_id: int, run_id: int) -> List[Experience]:
        self.env.reset(session=session_id)
        observation = self.env.observation
        self._env_done = False
        self._env_rounds = 0
        self._final_reward = 0.0
        self._session_id = session_id

        task_description = _extract_task_description(observation)
        history: List[str] = []
        memory = self.format_messages()
        turn_responses: List[Experience] = []
        turn_memories: List = []

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

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
            turn_memories.append(memory)

            await self._pre_student_gen(len(turn_memories) - 1, memory)

            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            if response.logprobs is None:
                raise RuntimeError(
                    "OnPolicyDistillVerlAgentWebshopWorkflow requires student model "
                    "to return logprobs. Set rollout_args.logprobs (e.g. 0) in task config."
                )
            turn_responses.append(response)

            await self._post_student_gen(len(turn_responses) - 1, response, memory)

            response = await self._select_env_response(
                len(turn_responses) - 1, response, memory
            )
            turn_responses[-1] = response
            response_text = response.response_text or ""

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


        return await self._score_and_finalize(turn_responses, turn_memories, run_id)

    async def _pre_student_gen(self, turn_index: int, memory) -> None:
        """Hook fired after the turn prompt is built, before the student samples."""
        return None

    async def _post_student_gen(
        self, turn_index: int, response: Experience, memory
    ) -> None:
        """Hook fired right after the student's turn Experience is collected."""
        return None

    async def _select_env_response(
        self, turn_index: int, response: Experience, memory
    ) -> Experience:
        """Select which turn response is actually executed in the environment."""
        return response

    async def _teacher_logprobs(self, response: Experience):
        """Full-sequence teacher log-probs for one turn (overridable seam).

        Args:
            response: the turn's Experience; ``response.tokens`` is the full
                ``[prompt | response]`` sequence the student saw.

        Returns:
            A ``[len(tokens) - 1]`` tensor of teacher log-probs, un-sliced.
        """
        return await self.teacher_model.logprobs_async(
            tokens=response.tokens.tolist(),
            temperature=self.temperature,
        )

    async def _score_and_finalize(
        self,
        turn_responses: List[Experience],
        turn_memories: Optional[List] = None,
        run_id: int = 0,
    ) -> List[Experience]:
        per_turn_kl_sums: List[float] = []
        for i, response in enumerate(turn_responses):
            teacher_logprobs = await self._teacher_logprobs(response)

            resp_start = response.prompt_length - 1
            teacher_resp_logprobs = teacher_logprobs[resp_start:]
            student_resp_logprobs = response.logprobs

            assert len(teacher_resp_logprobs) == len(student_resp_logprobs), (
                f"Length mismatch: teacher_logprobs={len(teacher_resp_logprobs)}, "
                f"student_logprobs={len(student_resp_logprobs)}. "
                f"tokens={len(response.tokens)}, prompt_length={response.prompt_length}"
            )

            response.teacher_logprobs = teacher_resp_logprobs

            if response.metrics is None:
                response.metrics = {}
            response.reward = self.compute_reward(response)
            response.eid.run = run_id
            response.eid.step = i

            kl_sum = (student_resp_logprobs - teacher_resp_logprobs).sum().item()
            per_turn_kl_sums.append(kl_sum)

        trajectory_kl_divergence = sum(per_turn_kl_sums)
        if turn_responses:
            last_response = turn_responses[-1]
            if last_response.metrics is None:
                last_response.metrics = {}
            last_response.metrics["env_rounds"] = self._env_rounds
            last_response.metrics["env_done"] = 1.0 if self._env_done else 0.0
            last_response.metrics["kl_divergence"] = trajectory_kl_divergence
            last_response.metrics["session_id"] = float(getattr(self, "_session_id", -1))

        return turn_responses
