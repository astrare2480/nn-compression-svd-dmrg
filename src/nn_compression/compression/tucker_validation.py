"""Tucker / HOOI 向けの入力 validation（アルゴリズム本体ではない）。

mode / rank / shape の contract を public API 境界で統一する。
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from .svd import coerce_integer_scalar, coerce_rank


def mode_unfold_max_rank(shape: tuple[int, ...], mode: int) -> int:
    """mode-n unfolding 上の truncated SVD で取りうる最大 rank。"""
    validate_mode_index(mode, len(shape), name="mode")
    row_dim = shape[mode]
    other_product = 1
    for index, dim in enumerate(shape):
        if index != mode:
            other_product *= dim
    return min(row_dim, other_product)


def validate_positive_shape(shape: tuple[int, ...]) -> None:
    """Tucker 対象 shape の各 dimension が正の整数であることを確認する。"""
    if not shape:
        raise ValueError("shape は空にできません。")
    if len(shape) < 2:
        raise ValueError(
            f"shape は2次元以上である必要があります: len={len(shape)}"
        )
    for index, dim in enumerate(shape):
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise TypeError(
                f"shape[{index}] は bool 以外の整数である必要があります: {dim!r}"
            )
        if dim <= 0:
            raise ValueError(
                f"shape[{index}] は正の整数である必要があります: {dim}"
            )


def validate_mode_index(mode: int, ndim: int, *, name: str = "mode") -> None:
    """mode が bool ではない整数かつ有効範囲内であることを確認する。"""
    if isinstance(mode, bool) or not isinstance(mode, int):
        raise TypeError(
            f"{name} は bool 以外の整数である必要があります: {mode!r}"
        )
    if not 0 <= mode < ndim:
        raise ValueError(
            f"{name}={mode} は 0〜{ndim - 1} の範囲で指定してください。"
        )


def validate_tucker_ranks(
    shape: tuple[int, ...],
    ranks: dict[int, int],
    *,
    allow_empty: bool = True,
) -> dict[int, int]:
    """Tucker rank 辞書を検証し、正規化した copy を返す。

    allow_empty=True のとき空 ranks ``{}`` のみを
    「圧縮しない identity 指定」として許可する。
    """
    if not isinstance(ranks, Mapping):
        raise TypeError(
            f"ranks は Mapping である必要があります: {type(ranks)!r}"
        )
    validate_positive_shape(shape)
    if len(ranks) == 0:
        if allow_empty:
            return {}
        raise ValueError("ranks は空にできません。")

    normalized: dict[int, int] = {}
    ndim = len(shape)
    for mode, rank in ranks.items():
        validate_mode_index(mode, ndim, name="mode")
        max_rank = mode_unfold_max_rank(shape, mode)
        normalized[mode] = coerce_rank(rank, max_rank, name=f"rank[{mode}]")
    return normalized


def validate_non_negative_tolerance(value: float, *, name: str) -> float:
    """abs_tol / rel_tol が有限かつ 0 以上であることを確認する。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} は数値である必要があります: {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} は有限値である必要があります: {value!r}")
    if numeric < 0:
        raise ValueError(f"{name} は 0 以上である必要があります: {numeric}")
    return numeric


def validate_max_iter(max_iter: int) -> int:
    """max_iter が bool ではない 0 以上の整数であることを確認する。"""
    max_iter_int = coerce_integer_scalar(max_iter, name="max_iter")
    if max_iter_int < 0:
        raise ValueError(
            f"max_iter は 0 以上である必要があります: {max_iter_int}"
        )
    return max_iter_int
