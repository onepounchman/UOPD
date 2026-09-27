# -*- coding: utf-8 -*-
"""Base Reward Function Class."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class RewardFn(ABC):
    """Base Reward Function Class."""

    @abstractmethod
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def __call__(self, **kwargs) -> Dict[str, float]:
        pass
