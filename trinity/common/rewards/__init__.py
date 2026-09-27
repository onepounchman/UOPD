"""Environment rewards are supplied by ALFWorld and WebShop."""
from trinity.common.rewards.reward_fn import RewardFn
from trinity.utils.registry import Registry

REWARD_FUNCTIONS = Registry("reward_functions", default_mapping={})
__all__ = ["RewardFn", "REWARD_FUNCTIONS"]
