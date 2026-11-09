"""文書単位の再標本化で、比の差の区間を出す。

同じ文書の抽選を、比べる方式のすべてに使う。比は

.. math::

    e = \\sum_d \\mathrm{numerator}_d / \\sum_d \\mathrm{denominator}_d

で、文書の平均の比ではない。

後半の ``bootstrap_counts`` からの関数は、同じ再標本を「各単位が何回選ばれたか」の
行列（counts）として持つ。行列を 1 度作れば、方式やセルをまたいで同じ再標本を
行列積で使い回せる。``bootstrap_ratio_difference_intervals`` と同じ ``seed`` なら、
選ばれる単位は同じ。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np


def bootstrap_ratio_difference_intervals(
    numerators: Mapping[str, np.ndarray],
    denominators: Mapping[str, np.ndarray],
    differences: Sequence[tuple[str, str]],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[tuple[str, str], dict[str, float]]:
    """方式ごとの分子・分母から、差の百分位区間を返す。

    numerators, denominators:
        方式名から shape ``(D,)`` の配列への対応。``D`` は文書数。
        同じ位置は同じ文書。
    differences:
        ``(left, right)`` の列。区間を出す量は ``e(left) - e(right)``。
    n_resamples, seed:
        再標本の回数と乱数の種。この実験では 1000 回、seed 0。
    confidence:
        区間の信頼度。0.95 なら 2.5% 点と 97.5% 点。

    戻り値:
        ``(left, right)`` から、点推定 ``estimate``、``low``、``high``、
        ``n_resamples`` への辞書。点推定は再標本ではなく、全文書の比の差。
        分母の合計が 0 の再標本があるときは ``ValueError``（その抽選を捨てない）。
    """
    if n_resamples < 1:
        raise ValueError(f"n_resamples は 1 以上である必要があります: {n_resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence は 0 と 1 の間である必要があります: {confidence}")
    names = list(numerators)
    if set(names) != set(denominators):
        raise ValueError("numerators と denominators の方式名が一致しません。")
    arrays_num = {}
    arrays_den = {}
    n_docs = None
    for name in names:
        num = np.asarray(numerators[name], dtype=np.float64)
        den = np.asarray(denominators[name], dtype=np.float64)
        if num.ndim != 1 or den.shape != num.shape:
            raise ValueError(f"{name} の分子・分母は同じ長さの 1 次元である必要があります。")
        if n_docs is None:
            n_docs = int(num.shape[0])
        elif num.shape[0] != n_docs:
            raise ValueError("方式ごとに文書数が違います。")
        arrays_num[name] = num
        arrays_den[name] = den
    if n_docs is None or n_docs < 1:
        raise ValueError("文書がありません。")
    for left, right in differences:
        if left not in arrays_num or right not in arrays_num:
            raise KeyError(f"差の方式名が見つかりません: {(left, right)}")

    def _ratio(num: np.ndarray, den: np.ndarray) -> float:
        denominator = float(den.sum())
        if denominator == 0.0:
            raise ValueError("分母の合計が 0 です。")
        return float(num.sum()) / denominator

    generator = np.random.default_rng(seed)
    draws = generator.integers(0, n_docs, size=(n_resamples, n_docs))
    resampled: dict[str, np.ndarray] = {}
    for name in names:
        num = arrays_num[name][draws].sum(axis=1)
        den = arrays_den[name][draws].sum(axis=1)
        if np.any(den == 0.0):
            raise ValueError(f"{name} の再標本で分母の合計が 0 になりました。")
        resampled[name] = num / den

    tail = (1.0 - confidence) / 2.0
    results: dict[tuple[str, str], dict[str, float]] = {}
    for left, right in differences:
        samples = resampled[left] - resampled[right]
        results[(left, right)] = {
            "estimate": _ratio(arrays_num[left], arrays_den[left])
            - _ratio(arrays_num[right], arrays_den[right]),
            "low": float(np.quantile(samples, tail)),
            "high": float(np.quantile(samples, 1.0 - tail)),
            "n_resamples": float(n_resamples),
        }
    return results


def bootstrap_counts(n_units: int, *, n_resamples: int, seed: int) -> np.ndarray:
    """再標本ごとに、各単位が何回選ばれたかの行列 ``(n_resamples, n_units)`` を返す。

    単位（記事・文書などのまとまり）を復元抽出で ``n_units`` 個選ぶ。行の和はどれも ``n_units``。
    乱数は ``numpy.random.default_rng(seed).integers`` で、``bootstrap_ratio_difference_intervals``
    と同じ抽選になる。
    """
    if n_units < 1 or n_resamples < 1:
        raise ValueError(f"n_units と n_resamples は 1 以上: {n_units}, {n_resamples}")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_units, size=(n_resamples, n_units))
    counts = np.zeros((n_resamples, n_units), dtype=np.float64)
    rows = np.repeat(np.arange(n_resamples), n_units)
    np.add.at(counts, (rows, picks.reshape(-1)), 1.0)
    return counts


def counts_digest(counts: np.ndarray) -> str:
    """再標本の行列の sha256。どの計算も同じ再標本を使ったことを記録で照合するために使う。"""
    return hashlib.sha256(np.ascontiguousarray(counts).tobytes()).hexdigest()


def percentile_interval(values: np.ndarray, percentiles: tuple[float, float] = (2.5, 97.5)) -> tuple[float, float]:
    """再標本の値の百分位区間。既定は 95% 区間（2.5% 点と 97.5% 点）。"""
    low, high = np.percentile(values, list(percentiles))
    return float(low), float(high)


def ratio_point_and_boot(
    numerator: np.ndarray, denominator: np.ndarray, counts: np.ndarray
) -> tuple[float, np.ndarray]:
    """単位ごとの分子・分母から、合計どうしの比の点推定と、再標本ごとの値を返す。

    比は :math:`\\sum_d n_d / \\sum_d m_d` で、単位ごとの比の平均ではない。
    ``counts`` は ``bootstrap_counts`` の行列。再標本ごとの値は ``(counts @ n) / (counts @ m)``。
    """
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    if numerator.shape != denominator.shape or numerator.ndim != 1:
        raise ValueError(f"形が違います: {numerator.shape} / {denominator.shape}")
    if counts.shape[1] != numerator.shape[0]:
        raise ValueError(f"単位の数が違います: {counts.shape[1]} != {numerator.shape[0]}")
    point = float(numerator.sum() / denominator.sum())
    boot = (counts @ numerator) / (counts @ denominator)
    return point, boot
