from trinity.algorithm.advantage_fn.advantage_fn import AdvantageFn, GroupAdvantage
from trinity.utils.registry import Registry

ADVANTAGE_FN: Registry = Registry(
    "advantage_fn",
    default_mapping={
        "on_policy_distill": "trinity.algorithm.advantage_fn.on_policy_distill_advantage.OnPolicyDistillAdvantage",
        "multi_turn_opd": "trinity.algorithm.advantage_fn.on_policy_distill_advantage.MultiTurnOpdAdvantage",
    },
)

__all__ = [
    "ADVANTAGE_FN",
    "AdvantageFn",
    "GroupAdvantage",
]
