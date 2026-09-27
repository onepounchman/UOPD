# -*- coding: utf-8 -*-
"""Rolling uncertainty scores and target intervention schedules for UOPD.

A shared Ray actor computes a quantile threshold from recent student-action
scores. Each episode retrieves one threshold before its rollout starts.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import ray

DEFAULT_ACTOR_NAME = "uopd_delta_window"


def intervention_rate(
    step: int,
    r_start: float,
    r_end: float,
    decay_steps: int,
    warmup_steps: int = 0,
    schedule: str = "linear",
) -> float:
    """Target intervention rate indexed by rollout batch, not optimizer step.

    Linear decay interpolates from r_start to r_end and holds r_end afterwards.
    A constant schedule holds r_start. Warmup, when enabled, delays decay.
    """
    if schedule == "constant":
        return r_start
    if schedule != "linear":
        raise ValueError(f"Unknown rate schedule: {schedule!r} (expected 'linear'/'constant')")
    if decay_steps <= warmup_steps:
        return r_end
    frac = (step - warmup_steps) / float(decay_steps - warmup_steps)
    frac = min(max(frac, 0.0), 1.0)
    return r_start + (r_end - r_start) * frac


@ray.remote(num_cpus=0)
class DeltaWindow:
    """Ring buffer of recently observed detector values, shared across runners.

    Holds plain floats rather than Experiences: every delta is measured on the *student's*
    proposed action (the teacher's decode never produces one), so nothing teacher-generated
    enters here even when ``teacher_takeover`` replaces the executed action.
    """

    def __init__(
        self,
        window_size: int = 1000,
        tau_abs: float = 0.0,
        n_min: int = 300,
        run_id: str = "",
    ) -> None:
        self.run_id = str(run_id)
        self._reset(window_size, tau_abs, n_min)

    def _reset(self, window_size: int, tau_abs: float, n_min: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if n_min < 0 or n_min > window_size:
            raise ValueError("n_min must be between 0 and window_size")
        self.buf = np.zeros(window_size, dtype=np.float32)
        self.window_size = window_size
        self.ptr = 0
        self.filled = 0
        # Not used for gating any more; kept as the reference the support diagnostic in
        # stats() is measured against (what fraction of observed turns are genuinely outside
        # the teacher's support, as opposed to merely in the worst r fraction of this batch).
        self.tau_abs = tau_abs
        self.n_min = n_min
        self.total_pushed = 0

    def ensure_run(
        self,
        run_id: str,
        window_size: int,
        tau_abs: float,
        n_min: int,
    ) -> bool:
        """Reset stale detached state when a new Ray job reuses this namespace.

        Returns ``True`` when a reset occurred. Repeated calls from runners in the
        same job preserve the shared window and verify that they agree on its shape.
        """
        run_id = str(run_id)
        if run_id != self.run_id:
            self.run_id = run_id
            self._reset(window_size, tau_abs, n_min)
            return True
        expected = (int(window_size), float(tau_abs), int(n_min))
        actual = (self.window_size, float(self.tau_abs), self.n_min)
        if expected != actual:
            raise ValueError(
                f"DeltaWindow configuration mismatch in run {run_id}: "
                f"existing={actual}, requested={expected}"
            )
        return False

    def push(self, deltas: List[float]) -> None:
        """Append one episode's deltas.

        Callers must push *every* turn's delta, triggered ones included. Triggered turns are by
        definition the high-delta tail; dropping them leaves only low values in the window, which
        biases the quantile downward, which triggers more turns, which drops more of the tail --
        a positive feedback that walks tau to zero.
        """
        if not deltas:
            return
        vals = np.asarray(deltas, dtype=np.float32)
        n = len(vals)
        if n >= self.window_size:  # a single episode longer than the window
            self.buf[:] = vals[-self.window_size :]
            self.ptr = 0
            self.filled = self.window_size
        else:
            end = self.ptr + n
            if end <= self.window_size:
                self.buf[self.ptr : end] = vals
            else:  # wrap
                split = self.window_size - self.ptr
                self.buf[self.ptr :] = vals[:split]
                self.buf[: end - self.window_size] = vals[split:]
            self.ptr = end % self.window_size
            self.filled = min(self.filled + n, self.window_size)
        self.total_pushed += n

    def tau(self, r: float) -> float:
        """Threshold for a target trigger rate ``r``.

        While the window is still cold this returns ``+inf``, i.e. nothing triggers and the
        first couple of steps run as plain OPD. That is deliberate rather than a fallback to
        some fixed threshold: the point is to seed the window with the *student's own* delta
        distribution. Forcing every turn instead would hand the environment to the teacher from
        turn 0, and the teacher's occupancy has a much lower detector value than the student's
        (a three- to four-fold gap in the entropy readings this method was built from), so the
        window would be seeded far too low and over-trigger for several steps after the switch.

        Filling the window costs on the order of one or two explore steps, during which the
        student is still at its initial checkpoint and has little to gain from correction.
        """
        if self.filled < self.n_min:
            return float("inf")
        r = min(max(r, 0.0), 1.0)
        if r == 0.0:
            # Disable intervention even if new scores exceed the window maximum.
            return float("inf")
        return float(np.quantile(self.buf[: self.filled], 1.0 - r))

    def rate_in_window(self, tau: float) -> float:
        """Fraction of the pooled deltas that this ``tau`` would fire on.

        Applying the returned ``tau`` back to the pool it came from separates the two reasons the
        realized rate can miss its target:

        * this value vs ``r`` is the **estimation** residual -- quantile discreteness and the
          strict ``delta > tau`` comparison dropping ties. It should be ~0 on continuous data.
        * the *episode's* rate vs ``r`` additionally carries the **distribution-shift** residual,
          since the pool is a lagging reference for a distribution that keeps drifting.

        Returns ``nan`` while the window is cold (no pool to measure against).
        """
        if not self.filled or not np.isfinite(tau):
            return float("nan")
        return float((self.buf[: self.filled] > tau).mean())

    def tau_and_rate(self, r: float) -> Tuple[float, float]:
        """``tau(r)`` plus the rate that ``tau`` actually fires on the pool, in ONE round-trip.

        The workflow needs both once per episode -- the threshold to gate with, and the pool-side
        rate that licenses the estimation residual -- and this keeps that to a single Ray call on
        the rollout path. ``tau`` is exactly what :meth:`tau` returns (``+inf`` while the window is
        cold), so gating behaviour is unchanged; the second value is ``nan`` while cold.
        """
        t = self.tau(r)
        return t, self.rate_in_window(t)

    def stats(self, r: Optional[float] = None) -> Dict[str, float]:
        """Diagnostics. ``frac_above_tau_abs`` is what licenses the support-test reading of the
        gate once the threshold itself has become a relative quantile."""
        out: Dict[str, float] = {
            "pool_size": float(self.filled),
            "total_pushed": float(self.total_pushed),
        }
        if self.filled:
            vals = self.buf[: self.filled]
            out["delta_mean"] = float(vals.mean())
            out["frac_above_tau_abs"] = float((vals > self.tau_abs).mean())
            if r is not None:
                t = self.tau(r)
                out["tau"] = t
                out["rate_in_window"] = self.rate_in_window(t)
                out["rate_est_err"] = self.rate_in_window(t) - r  # estimation residual only
        return out


def get_delta_window(
    name: str = DEFAULT_ACTOR_NAME,
    window_size: int = 1000,
    tau_abs: float = 0.0,
    n_min: int = 300,
):
    """Fetch (or create) the shared window actor.

    ``get_if_exists`` makes this race-free across the runners that all call it on their first
    episode; whichever gets there first creates it and the rest attach.
    """
    context = ray.get_runtime_context()
    run_id = str(context.get_job_id())
    actor = DeltaWindow.options(  # type: ignore[attr-defined]
        name=name,
        namespace=context.namespace,
        get_if_exists=True,
        lifetime="detached",
    ).remote(window_size=window_size, tau_abs=tau_abs, n_min=n_min, run_id=run_id)
    # The actor is detached so a previous driver may have left it behind. Every
    # runner in this job supplies the same job id; only the first call resets it.
    ray.get(actor.ensure_run.remote(run_id, window_size, tau_abs, n_min))
    return actor
