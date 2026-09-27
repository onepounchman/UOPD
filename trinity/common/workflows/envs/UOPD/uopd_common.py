# -*- coding: utf-8 -*-
"""Shared uncertainty scoring and teacher-action training targets for UOPD.

At an intervention turn, the teacher action is executed and its tokens become
an SFT target. Student-executed turns receive OPD supervision.
"""

import random as _random
from typing import Dict, List, Optional

import torch

from trinity.common.experience import CustomField, Experience

# Every turn Experience (both branches) must declare this identical custom_fields tuple,
# or to_data_proto() rejects the batch ("Custom fields are not consistent").
UOPD_TRIGGER_FIELD = CustomField(
    source_field="is_triggered",
    destination_field="is_triggered",
    data_type=torch.bool,
)


# Detector modes. All are length-normalized over the turn's response tokens and all are
# signed so that *larger means less familiar*, i.e. the trigger is always ``delta > tau``.
#
#   two-sided (compares student against teacher)
#     kl_k1      mean(log pi_S - log pi_T)   Schulman k1 estimator of D_KL(pi_S || pi_T)
#     kl_k3      mean(exp(-r) - 1 + r), r = log pi_S - log pi_T   (non-negative variant)
#   one-sided (reads the teacher alone, along the student's action)
#     t_entropy  mean H(pi_T(. | s_t, a_<i))     teacher's predictive entropy
#     confidence mean(-log pi_T(a_i | .))        teacher's surprisal at the student's tokens
#   uninformative (the ablation control)
#     random     U(0,1), independent of both models
#
# ``confidence`` negates the mean teacher log probability so larger scores indicate
# greater uncertainty.
#
# ``random`` is an uninformative control. With tau_mode="bernoulli", each turn draws
# u ~ U(0,1) and triggers iff u < r at the current scheduled rate. No window is used.
# The older tau_mode="quantile" variant estimates the cutoff from pooled random scores.
# Both match the target schedule; neither guarantees identical realized counts to UOPD.
DETECT_UQ_MODES = ("kl_k1", "kl_k3", "t_entropy", "confidence", "random")

# Superseded spellings kept so pre-rename run configs keep resolving.
_DETECT_UQ_ALIASES = {"k1": "kl_k1", "k3": "kl_k3"}

# Modes that need the teacher's per-token entropy rather than just its log-probs.
DETECT_UQ_NEEDS_ENTROPY = ("t_entropy",)

# Random selection supports the legacy quantile control and direct scheduled Bernoulli
# intervention (tau_mode="bernoulli"), which does not create a score window.
DETECT_UQ_UNINFORMATIVE = ("random",)

# Draw stream for ``detect_uq="random"``. Kept module-level and explicitly seedable so a control
# run is reproducible per process; note that with several async runners the *interleaving* is not
# deterministic, so what reproduces is the trigger RATE and its schedule, not the identity of the
# individual turns. That is the quantity the control exists to match.
_RANDOM_DETECTOR_RNG = _random.Random()
_RANDOM_DETECTOR_SEEDED = False


def seed_detect_uq_rng(seed: Optional[int]) -> None:
    """Seed the ``detect_uq="random"`` draw stream ONCE per process. ``None`` is a no-op.

    Idempotent on purpose, and this is the whole point rather than a nicety. Workflows are
    constructed per episode, so a seeding call that took effect every time would restart the same
    stream at the start of every episode: turn *i* would draw the same value in every episode, the
    "random" gate would fire at fixed turn indices, and the control would silently become
    "always correct turn 3" -- correlated with position, which is precisely the confound it exists
    to rule out. Seeding once per process keeps the draws i.i.d. across episodes while still
    pinning the stream a given worker starts from.
    """
    global _RANDOM_DETECTOR_SEEDED
    if seed is None or _RANDOM_DETECTOR_SEEDED:
        return
    _RANDOM_DETECTOR_RNG.seed(int(seed))
    _RANDOM_DETECTOR_SEEDED = True


def normalize_detect_uq(mode: str) -> str:
    """Resolve a detector name, accepting the pre-rename spellings."""
    m = _DETECT_UQ_ALIASES.get(mode, mode)
    if m not in DETECT_UQ_MODES:
        raise ValueError(f"Unknown detect_uq mode: {mode!r} (expected one of {DETECT_UQ_MODES})")
    return m


def compute_detect_uq(
    student_lp: torch.Tensor,
    teacher_lp: torch.Tensor,
    mode: str = "kl_k1",
    teacher_entropy: Optional[torch.Tensor] = None,
) -> float:
    """Uncertainty detector delta_t over a turn's response tokens; larger = less familiar.

    Args:
        student_lp: per-token ``log pi_S`` of the student's sampled tokens (response length).
        teacher_lp: per-token ``log pi_T`` of the *same* tokens (response length).
        mode: one of ``DETECT_UQ_MODES`` (pre-rename ``k1``/``k3`` also accepted).
        teacher_entropy: per-token ``H(pi_T)`` over the same positions. Required for
            ``t_entropy`` and ignored otherwise; obtaining it needs top-k teacher logprobs,
            so the workflow only requests it when the mode asks for it.
    """
    mode = normalize_detect_uq(mode)
    if mode == "random":
        return _RANDOM_DETECTOR_RNG.random()
    if student_lp.numel() == 0:
        return 0.0
    s = student_lp.float()
    t = teacher_lp.float()

    if mode == "kl_k1":
        return (s - t).mean().item()
    if mode == "kl_k3":
        # r = log(pi_S / pi_T); exp(-r) - 1 + r >= 0 with the same expectation as k1.
        r = torch.clamp(s - t, min=-20.0, max=20.0)
        return (torch.exp(-r) - 1.0 + r).mean().item()
    if mode == "confidence":
        # Teacher surprisal at the student's tokens. Low teacher log-prob (the study's
        # "confidence gate", which fired below the 25th percentile) becomes a high delta.
        return (-t).mean().item()
    if mode == "t_entropy":
        if teacher_entropy is None or teacher_entropy.numel() == 0:
            raise ValueError(
                "detect_uq='t_entropy' requires per-token teacher entropy; the workflow must "
                "score the turn with top-k logprobs enabled (see _score_turn)."
            )
        return teacher_entropy.float().mean().item()
    raise ValueError(f"Unhandled detect_uq mode: {mode!r}")


def cached_detect_uq(
    cache: Dict[int, float],
    turn_index: int,
    student_lp: torch.Tensor,
    teacher_lp: torch.Tensor,
    mode: str,
    teacher_entropy: Optional[torch.Tensor] = None,
) -> float:
    """Reuse the decision-time score at finalization, including a random draw."""
    if turn_index not in cache:
        cache[turn_index] = compute_detect_uq(student_lp, teacher_lp, mode, teacher_entropy)
    return cache[turn_index]


def should_trigger(
    delta: float,
    tau: float,
    num_trig: int,
    trigger_cap: Optional[int],
    current_step: int,
    sft_warmup_steps: int,
    is_eval: bool,
    *,
    random_rate: Optional[float] = None,
) -> bool:
    """Decide whether a turn becomes a trigger (teacher-SFT) turn.

    - Never trigger during eval.
    - During the SFT warm-up window (``current_step < sft_warmup_steps``) force trigger.
    - With random_rate, delta is a cached U(0,1) draw: trigger iff delta < random_rate.
    - Otherwise trigger iff ``delta > tau`` and the per-episode ``trigger_cap`` is not reached.
    """
    if is_eval:
        return False
    if trigger_cap is not None and num_trig >= trigger_cap:
        return False
    if current_step < sft_warmup_steps:
        return True
    if random_rate is not None:
        return delta < random_rate
    return delta > tau


def in_cold_start(current_step: int, sft_warmup_steps: int, is_eval: bool) -> bool:
    """Is this explore step inside the teacher cold-start (SFT warm-up) phase?

    Cold start is not a new mechanism: ``should_trigger`` already forces *every* turn to trigger
    while ``current_step < sft_warmup_steps``, so with the teacher also driving the environment
    those turns are exactly "teacher rollout + SFT on the teacher's action". This predicate names
    that phase, and **must stay bit-identical to should_trigger's warm-up test** -- if the two ever
    disagree on a step you get a turn where the teacher drives but the loss is OPD (or the reverse).

    STEP NUMBERING (bites in practice): the explorer schedules ``batch_id = explore_step_num + 1``
    with ``explore_step_num`` starting at 0 (``trinity/explorer/explorer.py``), so the first explore
    step is **1, not 0**. With the strict ``<`` inherited from ``should_trigger`` the phase therefore
    covers steps ``1 .. sft_warmup_steps - 1``, i.e. ``sft_warmup_steps - 1`` steps, and
    ``sft_warmup_steps=1`` is a no-op. Pass ``N+1`` to get ``N`` cold steps; the workflow asserts
    against the degenerate settings.

    Used for two things that must key off one definition:

    * the teacher drives the env even when ``teacher_takeover`` is off for the rest of the run
      (``cold_start_takeover``);
    * deltas from turns the teacher drove are **withheld from the quantile window** (see
      ``DeltaWindow.tau``): measured on the teacher's occupancy they read three- to four-fold lower,
      so pooling them sets the quantile far too low and over-triggers after the switch.

    Eval never cold-starts. ``sft_warmup_steps<=0`` (default) disables it entirely.
    """
    if is_eval or sft_warmup_steps <= 0:
        return False
    return current_step < sft_warmup_steps


def rate_tracking_error(realized: float, target: float) -> float:
    """Signed gap between the realized trigger rate and the rate the schedule asked for.

    The quantile gate sets ``tau`` so that a target fraction ``r`` of *recent, past* turns would
    have fired. What actually fires is measured on the *current* episode, and the two differ for
    two compounding reasons: the window is a lagging reference (the delta distribution keeps
    drifting down as the student trains, so a tau fitted on older, higher deltas under-fires on
    newer ones), and the strict ``delta > tau`` comparison drops ties at the cut.

    Negative means under-triggering, which is the direction observed in practice (~-0.04 on a
    250-step ALFWorld run). Logging it makes the gate auditable: a rate schedule is only
    meaningful if the realized rate tracks it, and this is the residual that says whether it does.
    """
    return float(realized) - float(target)


def mark_experience(exp: Experience, is_triggered: bool) -> Experience:
    """Tag a turn Experience with the UOPD per-row flag + custom field (both branches)."""
    if exp.info is None:
        exp.info = {}
    exp.info["is_triggered"] = bool(is_triggered)
    exp.custom_fields = [UOPD_TRIGGER_FIELD]
    return exp


def build_trigger_experience(teacher_turn: Experience) -> Experience:
    """Turn a teacher-decoded Experience into a UOPD trigger row.

    ``teacher_turn`` comes from ``teacher_model.chat_async`` on the same per-turn prompt, so its
    ``tokens = [prompt | teacher_response]``, ``logprobs`` are the teacher's response log-probs,
    and ``action_mask`` is ones over the response (set by the Experience constructor). The trainer
    recomputes ``log pi_student`` over ``tokens`` at SFT time, so no student scoring is needed here.

    We only add a length-consistent ``teacher_logprobs`` placeholder: the all-or-nothing batch
    gather drops the field unless *every* row has it, and SFT ignores its value.
    """
    resp_len = len(teacher_turn.tokens) - teacher_turn.prompt_length
    if teacher_turn.action_mask is None:
        teacher_turn.action_mask = torch.ones(resp_len, dtype=torch.bool)
    if teacher_turn.logprobs is not None:
        teacher_turn.teacher_logprobs = teacher_turn.logprobs.detach().clone()
    else:
        teacher_turn.teacher_logprobs = torch.zeros(resp_len, dtype=torch.float32)
    return mark_experience(teacher_turn, is_triggered=True)


def trigger_turn_metrics(triggered_turns: List[int], n_turns: int) -> Dict[str, float]:
    """Per-episode trigger summary.

    The metrics pipeline only keeps int/float and auto-derives mean/max/min per key, so a
    per-turn-index family (rate_00..rate_NN) explodes into ~3*N noisy curves whose high-index
    tails have tiny denominators. We instead emit summaries.

    Emitted for *every* episode, so their batch means are the episode-level view:

    - ``uopd/episode_intervened``: 1 if this episode was corrected at all. Its mean is the
      **intervention coverage**, i.e. the fraction of trajectories that received any teacher
      correction. This is a different quantity from ``uopd/trigger_rate``, which is the
      fraction of *turns*: a turn-level rate of r can be spread thinly over every episode or
      concentrated in a few, and only the pair distinguishes those.
    - ``uopd/trig_per_episode``: number of corrected turns in this episode.

    Emitted only when the episode triggered at least once, so zero-trigger episodes don't
    dilute them toward 0:

    - ``trigger_turn/mean_turn``: mean triggered turn index (absolute, e.g. 26).
    - ``trigger_turn/mean_frac``: mean *length-normalized* position ``t/(n_turns-1)`` in [0,1]
      (0 = episode start, 1 = end); comparable across episodes of different length.
    """
    metrics: Dict[str, float] = {
        "uopd/episode_intervened": 1.0 if triggered_turns else 0.0,
        "uopd/trig_per_episode": float(len(triggered_turns)),
    }
    if triggered_turns:
        metrics["trigger_turn/mean_turn"] = sum(triggered_turns) / len(triggered_turns)
        denom = max(n_turns - 1, 1)
        metrics["trigger_turn/mean_frac"] = sum(
            t / denom for t in triggered_turns
        ) / len(triggered_turns)
    return metrics
