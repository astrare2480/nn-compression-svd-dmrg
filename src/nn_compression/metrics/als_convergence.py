"""2 本の長い ALS を比べる事後の判定（03 の設計 §9）。

1 本の ALS をいつ止めるかは ``tt_mps.metric_tt_solver.ConvergenceStopRule`` が決める。
このモジュールは、止まった 2 本の停止理由と最終の目的値から、初期値への依存を分類する。
等価とみなす幅は呼び出し側が明示的に渡す（03 では 1e-3、つまり 0.1%）。

後半は、ALS の掃引の記録を表（CSV）の 1 行にする関数と、目的値が最終値に近づいた掃引を探す関数。
記録の型は ``tt_mps.metric_tt_solver`` のものだが、属性を読むだけで import はしない。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

ALS_STOP_REASONS = ("converged", "max_sweeps", "local_solver_unresolved")
INITIALIZATION_VERDICTS = (
    "local_solver_unresolved",
    "max_sweeps_unresolved",
    "converged_equivalent_objective",
    "converged_initialization_dependent",
)


@dataclass(frozen=True)
class InitializationDependence:
    """判定と、2 本の最終目的値の相対差 ``δ_Q``（判定によらず計算する）。"""

    verdict: str
    delta_q: float
    equivalence_tolerance: float


def relative_objective_difference(q_lambda_a: float, q_lambda_b: float) -> float:
    """``δ_Q = |Q_a − Q_b| / min(Q_a, Q_b)``。左右を入れ替えても同じ値。"""
    for name, value in (("q_lambda_a", q_lambda_a), ("q_lambda_b", q_lambda_b)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} は実数である必要があります: {value!r}")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} は正の有限値である必要があります: {value!r}")
    return abs(float(q_lambda_a) - float(q_lambda_b)) / min(float(q_lambda_a), float(q_lambda_b))


def classify_als_initialization_dependence(
    stop_reason_a: str,
    stop_reason_b: str,
    q_lambda_a: float,
    q_lambda_b: float,
    *,
    equivalence_tolerance: float,
) -> InitializationDependence:
    """2 本の停止理由と最終 Q_λ から §9 の判定を返す。

    優先順：
    1. どちらかが ``local_solver_unresolved`` → ``local_solver_unresolved``
    2. それ以外で、どちらかが ``max_sweeps`` → ``max_sweeps_unresolved``
    3. 両方が ``converged`` → ``δ_Q < equivalence_tolerance`` なら
       ``converged_equivalent_objective``、そうでなければ ``converged_initialization_dependent``
    4. それ以外（知らない停止理由）→ ``ValueError``
    """
    for name, reason in (("stop_reason_a", stop_reason_a), ("stop_reason_b", stop_reason_b)):
        if reason not in ALS_STOP_REASONS:
            raise ValueError(f"{name} が不明です: {reason!r}")
    if isinstance(equivalence_tolerance, bool) or not isinstance(equivalence_tolerance, (int, float)):
        raise TypeError("equivalence_tolerance は実数である必要があります。")
    if not math.isfinite(equivalence_tolerance) or equivalence_tolerance <= 0:
        raise ValueError("equivalence_tolerance は正の有限値である必要があります。")
    delta_q = relative_objective_difference(q_lambda_a, q_lambda_b)
    reasons = {stop_reason_a, stop_reason_b}
    if "local_solver_unresolved" in reasons:
        verdict = "local_solver_unresolved"
    elif "max_sweeps" in reasons:
        verdict = "max_sweeps_unresolved"
    elif reasons == {"converged"}:
        verdict = (
            "converged_equivalent_objective"
            if delta_q < equivalence_tolerance
            else "converged_initialization_dependent"
        )
    else:  # pragma: no cover - 上の検査で到達しない
        raise ValueError(f"想定外の停止理由の組み合わせです: {stop_reason_a!r}, {stop_reason_b!r}")
    return InitializationDependence(
        verdict=verdict, delta_q=delta_q, equivalence_tolerance=float(equivalence_tolerance)
    )


def study_sweep_row(record: Any) -> dict[str, Any]:
    """``StudySweepRecord``（``run_metric_tt_als_study`` の 1 掃引）を、CSV の 1 行にする。

    更新ごとの段の数から、2 段目・3 段目まで進んだ更新の数と 3 段目の残差、
    局所の目標に届かなかった更新（``local_target_missed``）の数も数える。
    列の並びは呼び出し側の fieldnames で決める。
    """
    stage2 = sum(len(update.stages) >= 2 for update in record.updates)
    stage3 = [update.stages[2].residual for update in record.updates if len(update.stages) >= 3]
    warnings = sum(update.outcome == "local_target_missed" for update in record.updates)
    return {
        "sweep": record.sweep,
        "q_lambda": record.q_lambda,
        "q_unregularized": record.q_unregularized,
        "relative_decrease": record.relative_decrease,
        "delta_50": record.delta_50,
        "consecutive_small_windows": record.consecutive_small_windows,
        "s_max": record.stationarity_max,
        "stationarity": ";".join(str(value) for value in record.stationarity),
        "n_updates": len(record.updates),
        "n_updates_stage2": stage2,
        "n_updates_stage3": len(stage3),
        "n_warnings_sweep": warnings,
        "n_warnings": record.n_warnings,
        "n_stage2": record.n_stage2,
        "n_stage3": record.n_stage3,
        "stage3_residuals": ";".join(str(value) for value in stage3),
        "hvp_calls": record.hvp_calls,
        "n_rejected": record.n_rejected,
        "seconds": record.seconds,
        "peak_memory_bytes": record.peak_memory_bytes,
        "stop_reason": record.stop_reason,
    }


def first_sweep_within(history: Sequence[Any], tolerance: float) -> int | None:
    """``Q_λ ≤ 最終の Q_λ × (1 + tolerance)`` に最初に入った掃引の番号。

    ``history`` は ``sweep`` と ``q_lambda`` を持つ記録の列（掃引順）。空なら ``None``。
    """
    if not history:
        return None
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError(f"tolerance は 0 以上の有限値です: {tolerance}")
    final = history[-1].q_lambda
    for record in history:
        if record.q_lambda <= final * (1.0 + tolerance):
            return record.sweep
    return None
