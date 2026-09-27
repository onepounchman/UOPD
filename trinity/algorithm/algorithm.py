# -*- coding: utf-8 -*-
"""Algorithm classes."""

from abc import ABC, ABCMeta, abstractmethod
from typing import Dict

from trinity.common.config import Config
from trinity.common.constants import SyncMethod
from trinity.utils.log import get_logger

logger = get_logger(__name__)


class ConstantMeta(ABCMeta):
    def __setattr__(cls, name, value):
        if name in cls.__dict__:
            raise AttributeError(f"{name} is already defined in {cls.__name__}")
        return super().__setattr__(name, value)


class AlgorithmType(ABC, metaclass=ConstantMeta):
    use_critic: bool  # whether to use critic model

    use_reference: bool  # whether to use reference model

    compute_advantage_in_trainer: bool  # whether to compute advantage in trainer
    # For algorithms that rely on experience grouping,
    # we recommend set this value to False

    can_balance_batch: bool  # balance batch in trainer

    schema: str  # schema of training data

    @classmethod
    @abstractmethod
    def default_config(cls) -> Dict:
        raise NotImplementedError

    @classmethod
    def name(cls) -> str:
        return cls._name

    @classmethod
    def check_config(cls, config: Config) -> None:
        pass


class OnPolicyDistillAlgorithm(AlgorithmType):
    """On-Policy Distillation Algorithm.

    Reference: Tinker library.

    Workflow stores teacher_logprobs in experience.info["teacher_logprobs"].
    Trainer's advantage_fn computes: advantages = teacher_logprobs - student_logprobs
    Trainer uses:
        importance_sampling loss if no clipping is needed
        ppo loss if clipping is needed, for better stability
    """

    use_critic: bool = False
    use_reference: bool = False
    compute_advantage_in_trainer: bool = True  # advantage_fn computes from teacher_logprobs
    can_balance_batch: bool = True
    schema: str = "experience"

    @classmethod
    def default_config(cls) -> Dict:
        return {
            "repeat_times": 1,
            "advantage_fn": "multi_turn_opd",
            "advantage_fn_args": {"kl_coef": 1.0},
            "sample_strategy": "default",
            "policy_loss_fn": "ppo",
            "kl_penalty_fn": "none",
            "kl_loss_fn": "none",
            "entropy_loss_fn": "none",
        }
