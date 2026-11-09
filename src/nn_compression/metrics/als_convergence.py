"""2 本の長い ALS を比べる事後の判定（03 の設計 §9）。

1 本の ALS をいつ止めるかは ``tt_mps.metric_tt_solver.ConvergenceStopRule`` が決める。
このモジュールは、止まった 2 本の停止理由と最終の目的値から、初期値への依存を分類する。
等価とみなす幅は呼び出し側が明示的に渡す（03 では 1e-3、つまり 0.1%）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

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
