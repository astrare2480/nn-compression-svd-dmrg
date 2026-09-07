"""TT-SVD 専用の入力 validation（アルゴリズム本体ではない）。

TT の内部 cut は ``1 <= k < d`` であり、汎用の mode index
（``nn_compression.tensor.validation.validate_mode_index`` の
``0 <= mode < ndim``）とは意味が異なる。Tucker/HOOI 専用の validation を
``tensor/validation.py`` に混ぜず ``compression/tucker_validation.py`` に
分離しているのと同じ理由で、TT 専用の contract もここに置く。

参照元:
    notebooks/30_tt_mps/00_fundamentals/01_tt_rank_unfolding.ipynb
"""

from __future__ import annotations


def validate_tt_cut_index(mode: int, ndim: int, *, name: str = "mode") -> None:
    """TT cut index の contract: ``1 <= mode < ndim``。

    TT の cut ``1:k | k+1:d`` は ``k=0``（左側が空）や ``k=d``（右側が空）
    を許さない。bool は整数のサブクラスだが意図しない誤用を避けるため
    明示的に拒否する。
    """
    if isinstance(mode, bool) or not isinstance(mode, int):
        raise TypeError(
            f"{name} は bool 以外の整数である必要があります: {mode!r}"
        )
    if not 1 <= mode < ndim:
        raise ValueError(
            f"{name}={mode} は 1〜{ndim - 1} の範囲で指定してください。"
        )
