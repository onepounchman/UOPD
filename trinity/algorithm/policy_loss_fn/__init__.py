from trinity.algorithm.policy_loss_fn.policy_loss_fn import PolicyLossFn
from trinity.utils.registry import Registry

POLICY_LOSS_FN: Registry = Registry(
    "policy_loss_fn",
    default_mapping={
        "ppo": "trinity.algorithm.policy_loss_fn.ppo_policy_loss.PPOPolicyLossFn",
        "uopd": "trinity.algorithm.policy_loss_fn.uopd_policy_loss.UOPDPolicyLossFn",
    },
)

__all__ = [
    "POLICY_LOSS_FN",
    "PolicyLossFn",
]
