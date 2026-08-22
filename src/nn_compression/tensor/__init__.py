"""mode-n テンソル演算（unfolding / folding / n-mode product）。"""

from .operations import fold, mode_dot, unfold

__all__ = [
    "unfold",
    "fold",
    "mode_dot",
]
