# -*- coding: utf-8 -*-
"""UOPD: reverse-KL policy gradients on student turns and SFT on teacher turns.

The per-row ``is_triggered`` field routes both branches through a single
token-mean aggregation, preserving the intervention-dependent loss weighting.
"""

from typing import Dict, Tuple

import torch

from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.algorithm.utils import aggregate_loss, masked_mean


class UOPDPolicyLossFn(PolicyLossFn):
    """Uncertainty-aware intervention distillation loss: per-turn OPD-PG / teacher-SFT mix."""

    def __init__(
        self,
        backend: str = "verl",
        loss_agg_mode: str = "token-mean",
        sft_coef: float = 1.0,
        sft_mode: str = "plain",
        clip_log_ratio: float = 20.0,
    ) -> None:
        super().__init__(backend=backend)
        self.loss_agg_mode = loss_agg_mode
        self.sft_coef = sft_coef
        if sft_mode != "plain":
            raise ValueError("UOPD uses plain teacher-action SFT (sft_mode=plain)")
        self.sft_mode = sft_mode
        self.clip_log_ratio = clip_log_ratio

    def __call__(  # type: ignore
        self,
        logprob: torch.Tensor,
        old_logprob: torch.Tensor,
        action_mask: torch.Tensor,
        advantages: torch.Tensor,
        is_triggered: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict]:
        # --- student branch: OPD reverse-KL PG, -A * ratio (discarded on trigger rows) ---
        log_ratio = torch.clamp(
            logprob - old_logprob, min=-self.clip_log_ratio, max=self.clip_log_ratio
        )
        ratio = torch.exp(log_ratio)
        pg_tok = -advantages * ratio

        # --- trigger branch: SFT on teacher tokens, sft_coef * w * (-logprob) ---
        sft_tok = self.sft_coef * (-logprob)

        # --- route per row (broadcast the [B] flag over the sequence dim) ---
        is_trig = is_triggered.bool().view(-1, 1)  # [B, 1]
        per_tok = torch.where(is_trig, sft_tok, pg_tok)
        loss = aggregate_loss(per_tok, action_mask, loss_agg_mode=self.loss_agg_mode)

        # --- metrics ---
        with torch.no_grad():
            trig_row = is_triggered.bool()
            trigger_rate = trig_row.float().mean().item() if trig_row.numel() > 0 else 0.0
            trig_tok_mask = action_mask * is_trig.to(action_mask.dtype)
            pg_tok_mask = action_mask * (~is_trig).to(action_mask.dtype)
            sft_loss_val = (
                masked_mean(sft_tok, trig_tok_mask).item() if trig_tok_mask.sum() > 0 else 0.0
            )
            pg_loss_val = (
                masked_mean(pg_tok, pg_tok_mask).item() if pg_tok_mask.sum() > 0 else 0.0
            )
        metrics = {
            "uopd/trigger_rate": trigger_rate,
            "uopd/sft_loss": sft_loss_val,
            "uopd/pg_loss": pg_loss_val,
            "uopd/sft_coef": float(self.sft_coef),
        }
        return loss, metrics

    @classmethod
    def default_args(cls) -> Dict:
        return {
            "loss_agg_mode": "token-mean",
            "sft_coef": 1.0,
            "sft_mode": "plain",
        }
