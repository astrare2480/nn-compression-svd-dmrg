"""本実験の評価の統計（設計 §7.3〜§7.7）。GPU もモデルも使わない。

単位（記事・文書）ごとに「誤差の二乗和」と「元の出力の二乗和」を持ち、
対になったクラスタのブートストラップで e・D・N・H_c の区間を作り、
セルを分類する。すべての方式とセルで、同じ再標本（同じ回数の行列）を使う。
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

N_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
THRESHOLDS = (0.2, 0.5)
MIN_DENOMINATOR_FRACTION = 0.05
W_B_F_MIN_CELLS = 26
MAIN_CELL_COUNT = 34

CLASS_ORDER = (
    "high",
    "mid",
    "low",
    "unresolved",
    "denominator_not_informative",
    "not_classified",
)


def bootstrap_counts(n_units: int, *, n_resamples: int = N_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """再標本ごとに、各単位が何回選ばれたかの行列 ``(n_resamples, n_units)`` を返す。

    単位を復元抽出で ``n_units`` 個選ぶ。行の和はどれも ``n_units``。
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
    """再標本の行列の sha256（すべての方式で同じものを使ったことの記録）。"""
    return hashlib.sha256(np.ascontiguousarray(counts).tobytes()).hexdigest()


# 同じ配列の sha256 を何度も計算しないための控え（id → (配列, sha256)）。
# 配列そのものも持つので、id が別の配列に使い回されることはない。
_DIGEST_CACHE: dict[int, tuple[np.ndarray, str]] = {}


def _used_counts_digest(counts: np.ndarray) -> str:
    """実際に計算に使った ``counts`` の sha256。結果の行に入れて、集計で照合する。"""
    cached = _DIGEST_CACHE.get(id(counts))
    if cached is not None and cached[0] is counts:
        return cached[1]
    digest = counts_digest(counts)
    _DIGEST_CACHE[id(counts)] = (counts, digest)
    return digest


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    """百分位の 95% 区間。"""
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def ratio_point_and_boot(
    numerator: np.ndarray, denominator: np.ndarray, counts: np.ndarray
) -> tuple[float, np.ndarray]:
    """単位ごとの和から、合計どうしの比（点推定と再標本ごとの値）を返す。

    単位ごとの比を平均しない（設計 §7.3）。
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


def _interval_dict(prefix: str, point: float, boot: np.ndarray) -> dict[str, float]:
    low, high = percentile_interval(boot)
    return {prefix: point, f"{prefix}_lo": low, f"{prefix}_hi": high}


def denominator_informative(*, d_point: float, d_lo: float, e3_point: float) -> bool:
    """分母 D = e(M3) − e(M2) が情報を持つか（設計 §7.4）。

    D が 0 以下、D の区間の下限が 0 以下、または D / e(M3) が 0.05 未満なら持たない。
    """
    return d_point > 0.0 and d_lo > 0.0 and d_point / e3_point >= MIN_DENOMINATOR_FRACTION


def classify_from_intervals(
    *,
    d_point: float,
    d_lo: float,
    e3_point: float,
    h02_lo: float,
    h02_hi: float,
    h05_lo: float,
    h05_hi: float,
    solver_unresolved: bool = False,
) -> str:
    """設計 §7.4 の表で分類する。

    ``not_classified``（M5 の局所ソルバーが解けなかった）を最初に、
    次に ``denominator_not_informative`` を当てはめる。
    D の区間が 0 をまたぐ ＝ D の点推定が正でも区間の下限が 0 以下。
    """
    if solver_unresolved:
        return "not_classified"
    if not denominator_informative(d_point=d_point, d_lo=d_lo, e3_point=e3_point):
        return "denominator_not_informative"
    if h05_lo >= 0.0:
        return "high"
    if h02_lo >= 0.0 and h05_hi < 0.0:
        return "mid"
    if h02_hi < 0.0:
        return "low"
    return "unresolved"


def evaluate_cell(
    numerators: Mapping[str, np.ndarray],
    denominator: np.ndarray,
    counts: np.ndarray,
    *,
    m2: str = "M2",
    m3: str = "M3",
    m5: str = "M5",
    m3f: str | None = "M3F",
    solver_unresolved: bool = False,
) -> dict[str, Any]:
    """1 セル・1 ドメインの e・D・N・H_c・R_M5 と分類を返す。

    ``numerators`` は方式名 → 単位ごとの誤差の二乗和。``denominator`` は単位ごとの
    元の出力の二乗和。すべての方式に同じ ``counts`` を使う。
    """
    for name in (m2, m3, m5):
        if name not in numerators:
            raise KeyError(f"{name} がありません。")
    e_point: dict[str, float] = {}
    e_boot: dict[str, np.ndarray] = {}
    for name, values in numerators.items():
        e_point[name], e_boot[name] = ratio_point_and_boot(values, denominator, counts)
    row: dict[str, Any] = {}
    for name in numerators:
        row.update(_interval_dict(f"e_{name}", e_point[name], e_boot[name]))
    d_boot = e_boot[m3] - e_boot[m2]
    n_boot = e_boot[m3] - e_boot[m5]
    d_point = e_point[m3] - e_point[m2]
    n_point = e_point[m3] - e_point[m5]
    row.update(_interval_dict("D", d_point, d_boot))
    row.update(_interval_dict("N", n_point, n_boot))
    for c in THRESHOLDS:
        label = f"H_{c:.1f}"
        row.update(_interval_dict(label, n_point - c * d_point, n_boot - c * d_boot))
    r_boot = e_boot[m5] - e_boot[m2]
    row.update(_interval_dict("R_M5", e_point[m5] - e_point[m2], r_boot))
    row["R_M5_over_e_M3"] = (e_point[m5] - e_point[m2]) / e_point[m3]
    row["kappa"] = n_point / e_point[m3]
    row["D_over_e_M3"] = d_point / e_point[m3]
    row["rho"] = n_point / d_point if d_point != 0.0 else math.nan
    if row["D_lo"] > 0.0:
        rho_lo, rho_hi = percentile_interval(n_boot / d_boot)
    else:
        rho_lo, rho_hi = math.nan, math.nan
    row["rho_lo"], row["rho_hi"] = rho_lo, rho_hi
    row["class"] = classify_from_intervals(
        d_point=d_point,
        d_lo=row["D_lo"],
        e3_point=e_point[m3],
        h02_lo=row["H_0.2_lo"],
        h02_hi=row["H_0.2_hi"],
        h05_lo=row["H_0.5_lo"],
        h05_hi=row["H_0.5_hi"],
        solver_unresolved=solver_unresolved,
    )
    # 1−ρ は分母が情報を持つときだけ出す（設計 §7.4）。
    informative = denominator_informative(d_point=d_point, d_lo=row["D_lo"], e3_point=e_point[m3])
    row["one_minus_rho"] = 1.0 - row["rho"] if informative else math.nan
    row["attained_residual_positive"] = bool(row["R_M5_lo"] > 0.0)
    row["counts_digest"] = _used_counts_digest(counts)
    if m3f is not None and m3f in numerators:
        n_f_boot = e_boot[m3] - e_boot[m3f]
        n_f_point = e_point[m3] - e_point[m3f]
        row.update(_interval_dict("H_0.2_F", n_f_point - 0.2 * d_point, n_f_boot - 0.2 * d_boot))
        row["rho_F"] = n_f_point / d_point if d_point != 0.0 else math.nan
        row["F_below_0.2"] = bool(row["H_0.2_F_hi"] < 0.0)
    return row


def calibration_rho(q_m2: float, q_m3: float, q_m5: float) -> float:
    """較正データの Q_λ での ρ（点推定だけ）。"""
    gap = q_m3 - q_m2
    return (q_m3 - q_m5) / gap if gap != 0.0 else math.nan


def aggregate_statements(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """設計 §7.5 の W・B・F を判定する。

    ``rows`` は主な集計の対象（確認用の multi_cut）のセルごとに 1 行で、
    ``class_wt2``・``class_c4``・``F_below_0.2_wt2`` を持つ。分母は 34 に固定する。
    """
    n = len(rows)
    if n != MAIN_CELL_COUNT:
        raise ValueError(f"主な集計のセル数が {n} です（{MAIN_CELL_COUNT} のはず）。")
    by_domain = {
        domain: {name: sum(1 for row in rows if row[f"class_{domain}"] == name) for name in CLASS_ORDER}
        for domain in ("wt2", "c4")
    }
    high_wt2 = by_domain["wt2"]["high"]
    low_wt2 = by_domain["wt2"]["low"]
    both_high = sum(1 for row in rows if row["class_wt2"] == "high" and row["class_c4"] == "high")
    any_low = sum(1 for row in rows if row["class_wt2"] == "low" or row["class_c4"] == "low")
    f_count = sum(1 for row in rows if bool(row["F_below_0.2_wt2"]))
    statement_w = high_wt2 >= W_B_F_MIN_CELLS and low_wt2 == 0
    statement_b = both_high >= W_B_F_MIN_CELLS and any_low == 0
    statement_f = statement_w and f_count >= W_B_F_MIN_CELLS
    return {
        "n_cells": n,
        "class_counts": by_domain,
        "high_wt2": high_wt2,
        "low_wt2": low_wt2,
        "high_both_domains": both_high,
        "low_any_domain": any_low,
        "m3f_below_0.2_wt2": f_count,
        "threshold": W_B_F_MIN_CELLS,
        "W": statement_w,
        "B": statement_b,
        "F": statement_f,
    }


def nll_report(
    nll_sums: Mapping[str, np.ndarray],
    n_tokens: np.ndarray,
    counts: np.ndarray,
    *,
    pairs: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """記事ごとの NLL の合計から、トークンあたりの NLL と差の区間を返す。

    各再標本で NLL = Σ(記事の NLL の合計) / Σ(記事のトークン数)。
    区間は差（``left − right``）にだけつける。PPL は表示用。
    """
    point: dict[str, float] = {}
    boot: dict[str, np.ndarray] = {}
    for name, values in nll_sums.items():
        point[name], boot[name] = ratio_point_and_boot(values, n_tokens, counts)
    out: dict[str, Any] = {}
    for name in nll_sums:
        out[f"nll_{name}"] = point[name]
        out[f"ppl_{name}"] = math.exp(point[name])
    for left, right in pairs:
        out.update(_interval_dict(f"dnll_{left}_minus_{right}", point[left] - point[right], boot[left] - boot[right]))
    out["counts_digest"] = _used_counts_digest(counts)
    return out
