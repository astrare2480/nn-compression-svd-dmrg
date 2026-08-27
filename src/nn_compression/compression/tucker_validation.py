"""Tucker / HOOI 向けの入力 validation（アルゴリズム本体ではない）。

mode / rank / shape の contract を public API 境界で統一する。
"""

from __future__ import annotations

import math
import operator

from .svd import coerce_rank


def validate_positive_shape(shape: tuple[int, ...]) -> None:
    """Tucker 対象 shape の各 dimension が正の整数であることを確認する。"""
    if not shape:
        raise ValueError("shape は空にできません。")
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

    allow_empty=True のとき空 ranks は「圧縮しない identity 指定」として許可する。
    """
    validate_positive_shape(shape)
    if not ranks:
        if allow_empty:
            return {}
        raise ValueError("ranks は空にできません。")

    normalized: dict[int, int] = {}
    ndim = len(shape)
    for mode, rank in ranks.items():
        validate_mode_index(mode, ndim, name="mode")
        normalized[mode] = coerce_rank(rank, shape[mode], name=f"rank[{mode}]")
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
    if isinstance(max_iter, bool):
        raise TypeError(f"max_iter は bool 以外の整数である必要があります: {max_iter!r}")
    try:
        max_iter_int = operator.index(max_iter)
    except TypeError as exc:
        raise TypeError(
            f"max_iter は整数である必要があります: {max_iter!r}"
        ) from exc
    if max_iter_int < 0:
        raise ValueError(
            f"max_iter は 0 以上である必要があります: {max_iter_int}"
        )
    return max_iter_int
