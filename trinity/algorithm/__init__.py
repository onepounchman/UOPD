from trinity.algorithm.advantage_fn import ADVANTAGE_FN, AdvantageFn
from trinity.algorithm.algorithm import AlgorithmType
from trinity.algorithm.entropy_loss_fn import ENTROPY_LOSS_FN, EntropyLossFn
from trinity.algorithm.kl_fn import KL_FN, KLFn
from trinity.algorithm.policy_loss_fn import POLICY_LOSS_FN, PolicyLossFn
from trinity.algorithm.sample_strategy import SAMPLE_STRATEGY, SampleStrategy
from trinity.utils.registry import Registry

ALGORITHM_TYPE: Registry = Registry(
    "algorithm",
    default_mapping={
        "on_policy_distill": "trinity.algorithm.algorithm.OnPolicyDistillAlgorithm",
    },
)


__all__ = [
    "ALGORITHM_TYPE",
    "AlgorithmType",
    "AdvantageFn",
    "ADVANTAGE_FN",
    "PolicyLossFn",
    "POLICY_LOSS_FN",
    "KLFn",
    "KL_FN",
    "EntropyLossFn",
    "ENTROPY_LOSS_FN",
    "SampleStrategy",
    "SAMPLE_STRATEGY",
]
