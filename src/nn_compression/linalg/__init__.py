"""モデル構造に依存しない線形代数。"""

from .svd import retained_energy_from_matrix, truncated_svd

__all__ = [
    "truncated_svd",
    "retained_energy_from_matrix",
]
