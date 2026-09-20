from .base import REGISTRY, Strategy, all_strategies, get_strategy, register
from . import builtin  # noqa: F401  导入即注册

__all__ = ["REGISTRY", "Strategy", "all_strategies", "get_strategy", "register"]
