# -*- coding: utf-8 -*-
"""On-Policy Distillation (OPD) workflow for AlfWorld.

Reference: OnPolicyDistillWorkflow logic; adapted for multi-turn AlfWorld.

Algorithm:
1. Student pre-samples trajectory (runs episode turn-by-turn with logprobs)
2. Split by turns: each turn has fixed prefix [system, obs_1, resp_1, ..., obs_t]
3. Teacher computes logprobs on same (prefix + response) per turn
4. Store teacher_logprobs in experience; advantage_fn uses teacher_logprobs - student_logprobs
5. Return one Experience per turn (like OPD returning one per sample)
"""

from dataclasses import asdict
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task, Workflow

from trinity.common.workflows.envs.UOPD.alfworld.utils import (
    ALFWORLD_TEMPLATE_NO_HIS,
    ALFWORLD_TEMPLATE,
    HISTORY_LENGTH,
    parse_action,
    format_observation,
    _extract_task,
    _format_history,
    _create_alfworld_env,
)

@WORKFLOWS.register_module("OPD_alfworld_workflow")
class OnPolicyDistillVerlAgentAlfworldWorkflow(Workflow):
    """On-policy distillation workflow for AlfWorld.

    Computes and stores teacher_logprobs in each turn's experience.
    The advantage_fn in trainer will compute:
        advantages = teacher_logprobs - student_logprobs

    Use advantage_fn: multi_turn_opd (MultiTurnOpdAdvantage) for this workflow,
    since it returns List[Experience] (one per turn), not a single response.

    Logic aligned with OnPolicyDistillWorkflow:
    - Student samples (with logprobs); teacher computes logprobs on same sequences.
    - Per-turn split: prefix fixed per turn, one Experience per turn.
    - compute_reward() can be overridden by subclasses (default: episode final reward).
    """

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
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )
        self.reset(task)

        # Eval is a pure student rollout (success rate); the teacher is not needed, so allow
        # benchmarking without any auxiliary model loaded. Training still requires the teacher.
        if self.is_eval:
            self.teacher_model = (
                self.auxiliary_model_wrappers[0] if self.auxiliary_model_wrappers else None
            )
        else:
            assert (
                self.auxiliary_model_wrappers is not None
                and len(self.auxiliary_model_wrappers) >= 1
            ), "On-policy distillation requires at least one auxiliary model as teacher."
            self.teacher_model = self.auxiliary_model_wrappers[0]

        self.temperature = task.workflow_args.get("temperature", 1.0)
        self.max_env_steps = task.workflow_args.get("max_env_steps", 50)
        self.is_eval = task.is_eval

    def reset(self, task: Task):
        """Reset the workflow with a new task.

        Unlike BaseSimpleWorkflow, this does NOT require reward_fn.
        """
        self.task = task
        self.format_args = task.format_args
        self.raw_task = task.raw_task
        self.task_desc = task.task_desc or "0"
        self.is_eval = task.is_eval

    def set_repeat_times(self, repeat_times, run_id_base):
        self.repeat_times = repeat_times
        self.task.rollout_args.n = repeat_times
        self.run_id_base = run_id_base

    def compute_reward(self, response: Experience) -> float:
        """Return episode-level reward (same for all turns in the trajectory).

        Set in _run_episode: env reward when done, 0.0 when max steps exhausted.
        """
        return getattr(self, "_final_reward", 0.0)

    @property
    def rollout_args(self):
        return asdict(self.task.rollout_args)

    def format_messages(self):
        """Format initial messages for the episode.

        Uses ALFWORLD_TEMPLATE_NO_HIS / ALFWORLD_TEMPLATE from utils.py.
        No system prompt; each user message is self-contained.
        """
        return []

    async def run_async(self) -> List[Experience]:
        game_file_path = self.task_desc
        env = _create_alfworld_env(game_file_path)
        try:
            return await self._run_episode(env)
        finally:
            env.close()

    async def _run_episode(self, env) -> List[Experience]:
        observation, info = env.reset()
        self._env_done = False
        self._env_rounds = 0

        task_description = _extract_task(observation)
        history: List[str] = []
        memory = self.format_messages()
        turn_responses: List[Experience] = []
        turn_memories: List = []

        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0

        for r in range(self.max_env_steps):
            format_obs = format_observation(observation)
            admissible_commands = info.get("admissible_commands", [])
            if admissible_commands and isinstance(admissible_commands[0], list):
                admissible_commands = admissible_commands[0]
            reformatted_admissible = "\n ".join(
                f"'{s}'" for s in admissible_commands if s != "help"
            )

            # NO_HIS only on the very first step (init), matching verl-agent
            # env_manager.build_text_obs (`if init`): ALFWORLD_TEMPLATE_NO_HIS
            # omits the task, so every step >= 1 must use ALFWORLD_TEMPLATE.
            if len(history) == 0:
                user_content = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=format_obs,
                    admissible_actions=reformatted_admissible,
                )
            else:
                action_history_str = "\n".join(
                    history[-HISTORY_LENGTH:]
                    if len(history) >= HISTORY_LENGTH
                    else history
                )
                user_content = ALFWORLD_TEMPLATE.format(
                    task_description=task_description,
                    step_count=r,
                    history_length=min(HISTORY_LENGTH, len(history)),
                    action_history=action_history_str,
                    current_step=r + 1,
                    current_observation=format_obs,
                    admissible_actions=reformatted_admissible,
                )

            memory = [{"role": "user", "content": user_content}]
            turn_memories.append(memory)

            # Hook (no-op in OPD): UOPD launches concurrent teacher work for this turn.
            await self._pre_student_gen(len(turn_memories) - 1, memory)

            # Step 1: Student samples this turn (same pattern as OnPolicyDistillWorkflow)
            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            if response.logprobs is None:
                raise RuntimeError(
                    "OnPolicyDistillAlfworldWorkflow requires student model to return logprobs. "
                    "Set rollout_args.logprobs (e.g. 0) in task config."
                )
            turn_responses.append(response)

            # Hook (no-op in OPD): UOPD scores + (pipeline) fires teacher decode inline.
            await self._post_student_gen(len(turn_responses) - 1, response, memory)

            # Hook (no-op in OPD): subclasses may replace the env-acting response for this turn.
            response = await self._select_env_response(
                len(turn_responses) - 1, response, memory
            )
            turn_responses[-1] = response
            response_text = response.response_text or ""

            action = parse_action(response_text)
            history.append(_format_history(format_obs, r + 1, action))
            observation, reward, done, info = env.step(action)
            if done:
                # done fires on success (won) OR on env truncation at the
                # internal step cap. Count success only when the task is won,
                # matching verl-agent (reward = 10 * info['won']).
                self._env_done = bool(info.get("won", False))
                self._env_rounds = r + 1
                self._final_reward = 1.0 if self._env_done else 0.0
                break
        else:
            self._env_rounds = self.max_env_steps
            self._final_reward = 0.0  # failure: exhausted max steps

        return await self._score_and_finalize(turn_responses, turn_memories)

    async def _pre_student_gen(self, turn_index: int, memory) -> None:
        """Hook fired after the turn prompt is built, before the student samples.

        No-op in OPD. UOPD overrides it to launch teacher work concurrently with the
        student generation (the ``eager`` schedule).
        """
        return None

    async def _post_student_gen(
        self, turn_index: int, response: Experience, memory
    ) -> None:
        """Hook fired right after the student's turn Experience is collected.

        No-op in OPD. UOPD overrides it to score + fire a conditional teacher decode
        inline during the rollout (the ``pipeline`` schedule).
        """
        return None

    async def _select_env_response(
        self, turn_index: int, response: Experience, memory
    ) -> Experience:
        """Select which turn response is actually executed in the environment.

        OPD keeps the student response. UOPD can override this to let a teacher-produced
        response take over the env action on selected turns.
        """
        return response

    async def _teacher_logprobs(self, response: Experience):
        """Compute teacher log probabilities for one student turn."""
        return await self.teacher_model.logprobs_async(
            tokens=response.tokens.tolist(), temperature=self.temperature,
        )

    async def _score_and_finalize(
        self, turn_responses: List[Experience], turn_memories: Optional[List] = None
    ) -> List[Experience]:
        # Eval: pure student rollout — no teacher needed. Record env metrics and return.
        if self.is_eval:
            if turn_responses:
                last = turn_responses[-1]
                if last.metrics is None:
                    last.metrics = {}
                last.metrics["env_rounds"] = self._env_rounds
                last.metrics["env_done"] = 1.0 if self._env_done else 0.0
            return turn_responses
        # Step 2 & 3: Teacher logprobs and fill experience (mirror OnPolicyDistillWorkflow.run_async)
        # response.tokens is the full sequence for this turn: [prefix | response], where
        # prefix = system + obs_1 + resp_1 + ... + obs_t (same input as student had).
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
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = i

            kl_sum = (student_resp_logprobs - teacher_resp_logprobs).sum().item()
            per_turn_kl_sums.append(kl_sum)

        # Trajectory-level metrics (computed once for the whole trajectory)
        trajectory_kl_divergence = sum(per_turn_kl_sums)
        if turn_responses:
            last_response = turn_responses[-1]
            if last_response.metrics is None:
                last_response.metrics = {}
            last_response.metrics["env_rounds"] = self._env_rounds
            last_response.metrics["env_done"] = 1.0 if self._env_done else 0.0
            last_response.metrics["kl_divergence"] = trajectory_kl_divergence

        return turn_responses
