# -*- coding: utf-8 -*-
"""UOPD workflow for webshop: execute teacher corrections and imitate them.

The default configuration uses teacher confidence, an adaptive quantile
threshold, and a decaying intervention budget. Student turns receive OPD
supervision; teacher-executed turns receive SFT supervision.
"""

import asyncio
import re
import time
from typing import List, Optional

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS, Task
from trinity.common.workflows.envs.UOPD.delta_window import (
    get_delta_window,
    intervention_rate,
)
from trinity.common.workflows.envs.UOPD.uopd_common import (
    build_trigger_experience,
    DETECT_UQ_NEEDS_ENTROPY,
    DETECT_UQ_UNINFORMATIVE,
    cached_detect_uq,
    normalize_detect_uq,
    seed_detect_uq_rng,
    mark_experience,
    in_cold_start,
    rate_tracking_error,
    should_trigger,
    trigger_turn_metrics,
)
from trinity.common.workflows.envs.UOPD.webshop.OPD_workflow import (
    OnPolicyDistillVerlAgentWebshopWorkflow,
)

_SCHED_MODES = ("lazy", "eager", "pipeline")


@WORKFLOWS.register_module("UOPD_webshop_workflow")
class UOPDWebshopWorkflow(OnPolicyDistillVerlAgentWebshopWorkflow):
    """UOPD workflow for WebShop (subclass of the OPD workflow)."""

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(task=task, model=model, auxiliary_models=auxiliary_models)
        self._load_uopd_settings(task.workflow_args or {})
        self._delta_window = None
        self._tau_episode = self.tau
        self._rate_in_window = float("nan")
        self._rate_now = 0.0

    def _load_uopd_settings(self, wa: dict) -> None:
        """Resolve and validate settings for the current task, including after reset."""
        self.detect_uq = normalize_detect_uq(wa.get("detect_uq", "confidence"))
        self.detect_needs_entropy = self.detect_uq in DETECT_UQ_NEEDS_ENTROPY
        self.detect_topk = wa.get("detect_topk", 20)
        self.tau = wa.get("tau", 0.0)
        self.detect_temperature = wa.get("detect_temperature", self.temperature)
        self.trigger_cap = wa.get("trigger_cap", None)
        self.sft_warmup_steps = wa.get("sft_warmup_steps", 0)
        self.teacher_sched = wa.get("teacher_sched", "pipeline")
        self.teacher_takeover = wa.get("teacher_takeover", True)
        # Threshold mode. "fixed" keeps tau constant (the original behaviour); "quantile"
        # recomputes it each episode as the 1-r quantile of recently observed deltas, so the
        # realized trigger rate follows the schedule below instead of drifting with the
        # detector distribution.
        self.tau_mode = wa.get("tau_mode", "quantile")
        assert self.tau_mode in ("fixed", "quantile", "bernoulli"), f"bad tau_mode: {self.tau_mode!r}"
        assert not (self.detect_uq in DETECT_UQ_UNINFORMATIVE and self.tau_mode not in ("quantile", "bernoulli")), (
            f"detect_uq={self.detect_uq!r} requires tau_mode='quantile' or 'bernoulli'"
        )
        assert self.tau_mode != "bernoulli" or self.detect_uq == "random", (
            "tau_mode='bernoulli' requires detect_uq='random'"
        )
        seed_detect_uq_rng(wa.get("detect_uq_seed", None))
        # "linear" decays r_start -> r_end by r_decay_steps; "constant" holds r_start
        # (the fixed-quantile baseline). tau_mode="fixed" is the fixed-threshold baseline.
        self.rate_schedule = wa.get("rate_schedule", "linear")
        assert self.rate_schedule in ("linear", "constant"), self.rate_schedule
        self.r_start = wa.get("r_start", 0.30)
        self.r_end = wa.get("r_end", 0.05)
        if self.tau_mode == "bernoulli":
            assert 0.0 <= self.r_start <= 1.0 and 0.0 <= self.r_end <= 1.0, (
                "Bernoulli intervention rates must be in [0, 1]"
            )
        self.r_decay_steps = wa.get("r_decay_steps", 120)
        self.window_size = wa.get("window_size", 1000)
        self.window_n_min = wa.get("window_n_min", 300)
        # Cold start: during the sft_warmup_steps prefix (where should_trigger already forces
        # every turn) let the TEACHER drive the env too, even if teacher_takeover is off for
        # the rest of the run. Gives clean teacher occupancy + SFT early, student exploration later.
        self.cold_start_takeover = wa.get("cold_start_takeover", False)
        # The phase is should_trigger's warm-up window, and the explorer's first batch_id is 1,
        # so `sft_warmup_steps` of 0 or 1 leaves NO cold step: the flag would be a silent no-op
        # while the run name and metrics claim a cold start happened.
        assert not (self.cold_start_takeover and self.sft_warmup_steps <= 1), (
            "cold_start_takeover=True needs sft_warmup_steps>=2 (the phase is steps "
            f"1..sft_warmup_steps-1); got sft_warmup_steps={self.sft_warmup_steps}"
        )
        assert self.teacher_sched in _SCHED_MODES, (
            f"teacher_sched must be one of {_SCHED_MODES} (got {self.teacher_sched!r})"
        )

    def _current_step(self) -> int:
        batch_id = self.task.batch_id
        if isinstance(batch_id, int):
            return batch_id
        if isinstance(batch_id, str):
            m = re.match(r"^(\d+)", batch_id)
            if m:
                return int(m.group(1))
        return 0

    def _teacher_decode_kwargs(self) -> dict:
        kwargs = {**self.rollout_args, "n": 1}
        if kwargs.get("logprobs") is None:
            kwargs["logprobs"] = 0
        return kwargs

    async def _refresh_tau(self) -> None:
        """Resolve this episode's threshold once, before any turn is generated.

        Pulled per episode rather than per turn: one RPC instead of one per turn, off the
        rollout critical path, and every turn of a trajectory is judged against the same
        threshold rather than a moving one.
        """
        if self.tau_mode == "bernoulli" and not self.is_eval:
            # Direct predefined probability: no score window or quantile RPC.
            self._rate_now = intervention_rate(
                self._current_step(), self.r_start, self.r_end,
                self.r_decay_steps, self.sft_warmup_steps, self.rate_schedule,
            )
            self._tau_episode = float("nan")
            self._rate_in_window = float("nan")
            return
        if self.tau_mode != "quantile" or self.is_eval:
            self._tau_episode = self.tau
            self._rate_in_window = float("nan")
            self._rate_now = 0.0
            return
        if self._delta_window is None:
            self._delta_window = get_delta_window(
                window_size=self.window_size, tau_abs=self.tau, n_min=self.window_n_min
            )
        self._rate_now = intervention_rate(
            self._current_step(),
            self.r_start,
            self.r_end,
            self.r_decay_steps,
            self.sft_warmup_steps,
            self.rate_schedule,
        )
        self._tau_episode, self._rate_in_window = await self._delta_window.tau_and_rate.remote(
            self._rate_now
        )

    async def _push_deltas(self, deltas) -> None:
        """Send this episode's deltas to the shared window (all turns, triggered included)."""
        if self.tau_mode != "quantile" or self.is_eval or not deltas:
            return
        if self._delta_window is None:
            return
        self._delta_window.push.remote([float(d) for d in deltas])

    async def _score_turn(self, response: Experience, turn_index: int = -1):
        """Teacher per-token logprobs for this turn (for delta).

        Entropy-based detectors additionally need top-k logprobs; we fetch them only in that
        case and cache the per-token entropy under ``turn_index``.
        """
        if self.detect_needs_entropy:
            teacher_logprobs, teacher_entropy = (
                await self.teacher_model.logprobs_with_entropy_async(
                    tokens=response.tokens.tolist(),
                    temperature=self.detect_temperature,
                    topk=self.detect_topk,
                )
            )
            self._teacher_ent[turn_index] = teacher_entropy[response.prompt_length - 1 :]
        else:
            teacher_logprobs = await self.teacher_model.logprobs_async(
                tokens=response.tokens.tolist(),
                temperature=self.detect_temperature,
            )
        resp_start = response.prompt_length - 1
        teacher_resp_lp = teacher_logprobs[resp_start:]

        assert len(teacher_resp_lp) == len(response.logprobs), (
            f"Length mismatch: teacher={len(teacher_resp_lp)}, student={len(response.logprobs)}, "
            f"tokens={len(response.tokens)}, prompt_length={response.prompt_length}"
        )
        return teacher_resp_lp

    async def run_async(self) -> List[Experience]:
        # A reset can replace the task without reconstructing this workflow.
        # Refresh UOPD settings before preparing the current rollout.
        self._load_uopd_settings(self.task.workflow_args or {})
        self._decode_futures = {}
        self._teacher_lp = {}
        self._teacher_ent = {}
        self._turn_deltas = {}
        await self._refresh_tau()
        self._trig_decisions = {}
        self._student_turns = {}
        self._teacher_turns = {}
        return await super().run_async()

    async def _pre_student_gen(self, turn_index: int, memory) -> None:
        if self.teacher_sched == "eager" and not self.is_eval:
            self._decode_futures[turn_index] = asyncio.ensure_future(
                self.teacher_model.chat_async(memory, **self._teacher_decode_kwargs())
            )

    async def _post_student_gen(self, turn_index: int, response: Experience, memory) -> None:
        if self.teacher_sched != "pipeline" or self.is_eval:
            return
        teacher_lp = await self._score_turn(response, turn_index)
        self._teacher_lp[turn_index] = teacher_lp
        delta = cached_detect_uq(self._turn_deltas, turn_index, response.logprobs, teacher_lp, self.detect_uq, self._teacher_ent.get(turn_index))
        do_trig = should_trigger(
            delta,
            self._tau_episode,
            sum(self._trig_decisions.values()),
            self.trigger_cap,
            self._current_step(),
            self.sft_warmup_steps,
            self.is_eval,
            random_rate=self._rate_now if self.tau_mode == "bernoulli" else None,
        )
        self._trig_decisions[turn_index] = do_trig
        if do_trig:
            self._decode_futures[turn_index] = asyncio.ensure_future(
                self.teacher_model.chat_async(memory, **self._teacher_decode_kwargs())
            )

    async def _select_env_response(
        self, turn_index: int, response: Experience, memory
    ) -> Experience:
        cold = in_cold_start(self._current_step(), self.sft_warmup_steps, self.is_eval)
        if not (self.teacher_takeover or (cold and self.cold_start_takeover)) or self.is_eval:
            return response

        teacher_lp = self._teacher_lp.get(turn_index)
        do_trig = self._trig_decisions.get(turn_index)
        if do_trig is None:
            if teacher_lp is None:
                teacher_lp = await self._score_turn(response, turn_index)
                self._teacher_lp[turn_index] = teacher_lp
            delta = cached_detect_uq(self._turn_deltas, turn_index, response.logprobs, teacher_lp, self.detect_uq, self._teacher_ent.get(turn_index))
            do_trig = should_trigger(
                delta,
                self._tau_episode,
                sum(self._trig_decisions.values()),
                self.trigger_cap,
                self._current_step(),
                self.sft_warmup_steps,
                self.is_eval,
                random_rate=self._rate_now if self.tau_mode == "bernoulli" else None,
            )
            self._trig_decisions[turn_index] = do_trig
        if not do_trig:
            return response

        teacher_turn = await self._get_teacher_turn(turn_index, memory)
        row = build_trigger_experience(teacher_turn)
        row.eid = response.eid
        self._student_turns[turn_index] = response
        self._teacher_turns[turn_index] = row
        return row

    async def _get_teacher_turn(self, turn_index: int, memory) -> Experience:
        fut = self._decode_futures.get(turn_index)
        if fut is not None:
            return (await fut)[0]
        return (
            await self.teacher_model.chat_async(memory, **self._teacher_decode_kwargs())
        )[0]

    async def _score_and_finalize(
        self,
        turn_responses: List[Experience],
        turn_memories: Optional[List] = None,
        run_id: int = 0,
    ) -> List[Experience]:
        if self.is_eval:
            if turn_responses:
                last = turn_responses[-1]
                if last.metrics is None:
                    last.metrics = {}
                last.metrics["env_rounds"] = self._env_rounds
                last.metrics["env_done"] = 1.0 if self._env_done else 0.0
                last.metrics["num_turns"] = float(len(turn_responses))
                last.metrics["session_id"] = float(getattr(self, "_session_id", -1))
            return turn_responses

        t0 = time.perf_counter()
        run = run_id
        current_step = self._current_step()
        num_trig = 0
        triggered_turns: List[int] = []
        per_turn_kl_sums: List[float] = []
        per_turn_deltas: List[float] = []

        for i, response in enumerate(turn_responses):
            student_response = self._student_turns.get(i, response)
            if i in self._trig_decisions:
                do_trig = self._trig_decisions[i]
                teacher_resp_lp = self._teacher_lp.get(i)
                if teacher_resp_lp is None:
                    teacher_resp_lp = await self._score_turn(student_response, i)
                    self._teacher_lp[i] = teacher_resp_lp
                delta = cached_detect_uq(self._turn_deltas, i, student_response.logprobs, teacher_resp_lp, self.detect_uq, self._teacher_ent.get(i))
            elif self.teacher_sched == "pipeline":
                teacher_resp_lp = self._teacher_lp.get(i)
                if teacher_resp_lp is None:
                    teacher_resp_lp = await self._score_turn(student_response, i)
                    self._teacher_lp[i] = teacher_resp_lp
                delta = cached_detect_uq(self._turn_deltas, i, student_response.logprobs, teacher_resp_lp, self.detect_uq, self._teacher_ent.get(i))
                do_trig = self._trig_decisions.get(i, False)
            else:
                teacher_resp_lp = await self._score_turn(student_response, i)
                self._teacher_lp[i] = teacher_resp_lp
                delta = cached_detect_uq(self._turn_deltas, i, student_response.logprobs, teacher_resp_lp, self.detect_uq, self._teacher_ent.get(i))
                do_trig = should_trigger(
                    delta,
                    self._tau_episode,
                    num_trig,
                    self.trigger_cap,
                    current_step,
                    self.sft_warmup_steps,
                    self.is_eval,
                    random_rate=self._rate_now if self.tau_mode == "bernoulli" else None,
                )
                self._trig_decisions[i] = do_trig
            student_resp_lp = student_response.logprobs
            per_turn_deltas.append(delta)

            if do_trig and turn_memories is not None:
                row = self._teacher_turns.get(i)
                if row is None:
                    teacher_turn = await self._get_teacher_turn(i, turn_memories[i])
                    row = build_trigger_experience(teacher_turn)
                    row.eid = response.eid
                    self._teacher_turns[i] = row
                row.eid = response.eid
                row.eid.run = run
                row.eid.step = i
                row.reward = self.compute_reward(student_response)
                turn_responses[i] = row
                num_trig += 1
                triggered_turns.append(i)
            else:
                response.teacher_logprobs = teacher_resp_lp
                response.reward = self.compute_reward(student_response)
                response.eid.run = run
                response.eid.step = i
                mark_experience(response, is_triggered=False)
                per_turn_kl_sums.append((student_resp_lp - teacher_resp_lp).sum().item())

        for fut in self._decode_futures.values():
            if not fut.done():
                try:
                    await fut
                except Exception:
                    pass

        if turn_responses:
            last = turn_responses[-1]
            if last.metrics is None:
                last.metrics = {}
            n_turns = len(turn_responses)
            last.metrics["env_rounds"] = self._env_rounds
            last.metrics["env_done"] = 1.0 if self._env_done else 0.0
            last.metrics["kl_divergence"] = sum(per_turn_kl_sums)
            last.metrics["num_triggers"] = float(num_trig)
            last.metrics["num_turns"] = float(n_turns)
            last.metrics["trigger_rate"] = num_trig / max(n_turns, 1)
            # The teacher also drives during a cold-start prefix, so these must not key on
            # teacher_takeover alone or the prefix reports "student drove everything".
            drove = self.teacher_takeover or (
                self.cold_start_takeover
                and in_cold_start(current_step, self.sft_warmup_steps, self.is_eval)
            )
            last.metrics["teacher_takeover"] = 1.0 if drove else 0.0
            last.metrics["teacher_env_rounds"] = float(num_trig if drove else 0)
            last.metrics["student_env_rounds"] = float(
                n_turns - (num_trig if drove else 0)
            )
            last.metrics["detect_uq"] = sum(per_turn_deltas) / max(len(per_turn_deltas), 1)
            if self.tau_mode != "bernoulli":
                last.metrics["uopd/tau"] = float(self._tau_episode)
            else:
                last.metrics["uopd/random_intervention"] = 1.0
            last.metrics["uopd/rate_target"] = float(self._rate_now)
            # Rate residuals apply to quantile/Bernoulli gating OUTSIDE the warm-up
            # prefix: in fixed mode the "target" is an unset 0.0, and during cold start every
            # turn is forced so the residual is ~1-r by construction, which would swamp the
            # batch mean rather than measure tracking.
            cold_now = in_cold_start(current_step, self.sft_warmup_steps, self.is_eval)
            if self.tau_mode in ("quantile", "bernoulli") and not cold_now:
                last.metrics["uopd/rate_err"] = rate_tracking_error(
                    num_trig / max(n_turns, 1), self._rate_now
                )
            # pool-side rate this tau fires on: isolates quantile-estimation error from the
            # distribution shift between the (lagging) pool and the current episode.
            if self.tau_mode == "quantile" and not cold_now and (
                self._rate_in_window == self._rate_in_window  # not nan
            ):
                last.metrics["uopd/rate_in_window"] = float(self._rate_in_window)
                last.metrics["uopd/rate_est_err"] = rate_tracking_error(
                    self._rate_in_window, self._rate_now
                )
            last.metrics["uopd/cold_start"] = 1.0 if cold_now else 0.0
            last.metrics["uopd/t_finalize"] = time.perf_counter() - t0
            last.metrics["session_id"] = float(getattr(self, "_session_id", -1))
            last.metrics.update(trigger_turn_metrics(triggered_turns, n_turns))

        # Feed the shared window: every turn's delta, triggered ones included, all
        # measured on the student's proposed action (see _push_deltas) -- EXCEPT turns the
        # teacher actually drove during a cold-start prefix. Those sit on the teacher's
        # occupancy, where the detector reads three- to four-fold lower, and pooling them sets
        # the quantile far too low (see DeltaWindow.tau). Filtered per TURN rather than per
        # episode, because trigger_cap can end the teacher's run part-way through one.
        pushable = per_turn_deltas
        if self.cold_start_takeover and in_cold_start(
            current_step, self.sft_warmup_steps, self.is_eval
        ):
            pushable = [
                d for i, d in enumerate(per_turn_deltas) if not self._trig_decisions.get(i, False)
            ]
        await self._push_deltas(pushable)

        return turn_responses
