# -*- coding: utf-8 -*-
"""Teacher-free evaluation workflow for AlfWorld (TCOD paper Table 2).

This reuses the EXACT rollout loop / prompt templates / action parsing /
success criterion of ``OPD_alfworld_workflow`` (so eval numbers are directly
comparable to the OPD training setup), but:

  * does NOT require a teacher / auxiliary model (no distillation here), and
  * skips the teacher-logprob / KL computation entirely.

It only measures, per episode, the two quantities reported in the paper:

  * ``env_done``   -> 1.0 if the task was solved, else 0.0   (=> Success Rate)
  * ``env_rounds`` -> number of interaction rounds            (=> Rounds)

The explorer aggregates these across an eval taskset as
``<prefix>/<taskset>/env_done/mean@1`` and ``.../env_rounds/mean@1``.

Use it via ``default_eval_workflow_type: 'eval_alfworld_workflow'`` in a
``mode: bench`` config (see TCOD/eval/).
"""

from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task, Workflow
from trinity.common.workflows.envs.UOPD.alfworld.OPD_workflow import (
    OnPolicyDistillVerlAgentAlfworldWorkflow,
)
from trinity.common.workflows.envs.UOPD.alfworld.utils import (
    ALFWORLD_TEMPLATE,
    ALFWORLD_TEMPLATE_NO_HIS,
    HISTORY_LENGTH,
    _create_alfworld_env,
    _extract_task,
    _format_history,
    format_observation,
    parse_action,
)


@WORKFLOWS.register_module("eval_alfworld_workflow")
class EvalAlfworldWorkflow(OnPolicyDistillVerlAgentAlfworldWorkflow):
    """Single-model AlfWorld rollout for evaluation (no teacher required)."""

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
        self.max_env_steps = task.workflow_args.get("max_env_steps", 50)
        self.is_eval = task.is_eval

    async def _run_episode(self, env) -> List[Experience]:
        observation, info = env.reset()
        self._env_done = False
        self._env_rounds = 0

        task_description = _extract_task(observation)
        history: List[str] = []
        memory = self.format_messages()
        turn_responses: List[Experience] = []

        kwargs = {**self.rollout_args, "n": 1}

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
                    history[-HISTORY_LENGTH:] if len(history) >= HISTORY_LENGTH else history
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

            responses = await self.model.chat_async(memory, **kwargs)
            response = responses[0]
            response_text = response.response_text or ""
            turn_responses.append(response)

            action = parse_action(response_text)
            history.append(_format_history(format_obs, r + 1, action))
            observation, reward, done, info = env.step(action)
            self._last_obs = format_observation(observation)
            self._last_action = action
            self._last_reward = reward
            self._last_won = bool(info.get("won", False))
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

        # Optional per-episode dump for auditing (set ALF_EVAL_DUMP=/path.tsv).
        # Columns: game | won | rounds | last_action | last_env_reward | last_obs(trunc)
        import os as _os
        # Optional full task identity audit, used to verify multi-split evaluations.
        _audit = _os.environ.get("ALF_EVAL_AUDIT")
        if _audit:
            import json as _json
            _record = _json.dumps({
                "game_file": str(self.task_desc),
                "won": bool(self._env_done),
                "rounds": int(self._env_rounds),
            }) + "\n"
            _fd = _os.open(_audit, _os.O_WRONLY | _os.O_CREAT | _os.O_APPEND, 0o644)
            try:
                _os.write(_fd, _record.encode())
            finally:
                _os.close(_fd)
        _dump = _os.environ.get("ALF_EVAL_DUMP")
        if _dump:
            try:
                _game = _os.path.basename(_os.path.dirname(_os.path.dirname(str(self.task_desc))))
                _obs = " ".join(getattr(self, "_last_obs", "").split())[:400]
                with open(_dump, "a") as _f:
                    _f.write(
                        f"{_game}\t{int(self._env_done)}\t{self._env_rounds}\t"
                        f"{getattr(self, '_last_action', '')}\t"
                        f"{getattr(self, '_last_reward', '')}\t{_obs}\n"
                    )
            except Exception as _e:
                print(f"[ALF_EVAL_DUMP] write failed: {_e}")

        # Finalize experiences (mirror OPD, minus teacher-logprob / KL).
        for i, response in enumerate(turn_responses):
            if response.metrics is None:
                response.metrics = {}
            response.reward = self.compute_reward(response)
            response.eid.run = getattr(self, "run_id_base", 0)
            response.eid.step = i

        if turn_responses:
            last_response = turn_responses[-1]
            if last_response.metrics is None:
                last_response.metrics = {}
            last_response.metrics["env_rounds"] = self._env_rounds
            last_response.metrics["env_done"] = 1.0 if self._env_done else 0.0

        return turn_responses
