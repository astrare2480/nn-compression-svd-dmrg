"""設計 §9 の ρ と、予備実験の進退判定。

結果を見てからしきい値を変えない。このモジュールは判定の式だけを持ち、
測定値は引数で受け取る。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RhoEvaluation:
    """1 つの条件（予算・tensorization・データ）の ρ。

    rho_status:
        ``ok`` または ``denominator_not_informative``。
    g_ci_low, g_ci_high:
        ``g = e(M3) - e(M2)`` の 95% 区間。区間が 0 をまたぐと ρ は計算しない。
    """

    budget: str
    tensorization: str
    domain: str
    e_m2: float
    e_m3: float
    e_m5: float
    g_ci_low: float
    g_ci_high: float
    g: float
    rho: float | None
    rho_status: str
    delta_tt: float
    r_m5: float
    kappa: float | None


def evaluate_rho(
    *,
    budget: str,
    tensorization: str,
    domain: str,
    e_m2: float,
    e_m3: float,
    e_m5: float,
    g_ci_low: float,
    g_ci_high: float,
    min_relative_g: float = 0.05,
) -> RhoEvaluation:
    """ρ と、ρ を出さないときの補助量を計算する。

    ``g <= 0``、``g`` の区間が 0 をまたぐ、または ``g / e(M3) < min_relative_g``
    のときは ``rho_status = denominator_not_informative`` とし、ρ は ``None``。
    その場合も ``delta_tt``、``r_m5``、``kappa`` は返す。``e(M3)=0`` の
    ``kappa`` だけ ``None``。
    """
    g_value = e_m3 - e_m2
    delta_tt = e_m3 - e_m5
    r_m5 = e_m5 - e_m2
    kappa = None if e_m3 == 0.0 else delta_tt / e_m3
    interval_crosses_zero = g_ci_low <= 0.0 <= g_ci_high
    relative_gap_small = e_m3 == 0.0 or (g_value / e_m3) < min_relative_g
    if g_value <= 0.0 or interval_crosses_zero or relative_gap_small:
        status = "denominator_not_informative"
        rho = None
    else:
        status = "ok"
        rho = delta_tt / g_value
    return RhoEvaluation(
        budget=budget,
        tensorization=tensorization,
        domain=domain,
        e_m2=e_m2,
        e_m3=e_m3,
        e_m5=e_m5,
        g_ci_low=g_ci_low,
        g_ci_high=g_ci_high,
        g=g_value,
        rho=rho,
        rho_status=status,
        delta_tt=delta_tt,
        r_m5=r_m5,
        kappa=kappa,
    )


def classify_pilot_outcome(
    cells: Sequence[RhoEvaluation],
    *,
    tensorizations: Sequence[str],
    budgets: Sequence[str],
    validation_domains: Sequence[str] = ("validation_same", "validation_other"),
    calibration_domain: str = "calibration",
    low: float = 0.2,
    high: float = 0.5,
) -> str:
    """設計 §9 の 4 つの判定のどれかを返す。

    戻り値は ``本実験へ進む``、``過適合が主な結果``、``中止``、``判定保留``
    のいずれか。期待する条件のセルが欠けているときは ``判定保留``。

    過適合は、中止の条件に加えて較正の ρ がすべて ``high`` 以上である場合。
    両方に当てはまるときは過適合を返す。本実験へ進む条件は、同じ
    tensorization が両方の予算と両方の検証ドメインで ρ >= ``high`` であること。
    """
    index = {(cell.tensorization, cell.budget, cell.domain): cell for cell in cells}

    def _cell(tensorization: str, budget: str, domain: str) -> RhoEvaluation | None:
        return index.get((tensorization, budget, domain))

    def _complete(domains: Sequence[str]) -> bool:
        return all(
            _cell(tensorization, budget, domain) is not None
            for tensorization in tensorizations
            for budget in budgets
            for domain in domains
        )

    if not _complete(tuple(validation_domains) + (calibration_domain,)):
        return "判定保留"

    def _ok_at_least(cell: RhoEvaluation, threshold: float) -> bool:
        return cell.rho_status == "ok" and cell.rho is not None and cell.rho >= threshold

    def _ok_below(cell: RhoEvaluation, threshold: float) -> bool:
        return cell.rho_status == "ok" and cell.rho is not None and cell.rho < threshold

    def _validation_cells() -> list[RhoEvaluation]:
        return [
            _cell(tensorization, budget, domain)
            for tensorization in tensorizations
            for budget in budgets
            for domain in validation_domains
            if _cell(tensorization, budget, domain) is not None
        ]

    def _calibration_cells() -> list[RhoEvaluation]:
        return [
            _cell(tensorization, budget, calibration_domain)
            for tensorization in tensorizations
            for budget in budgets
            if _cell(tensorization, budget, calibration_domain) is not None
        ]

    advances = False
    for tensorization in tensorizations:
        if all(
            _ok_at_least(_cell(tensorization, budget, domain), high)
            for budget in budgets
            for domain in validation_domains
        ):
            advances = True
            break
    if advances:
        return "本実験へ進む"

    validation = _validation_cells()
    calibration = _calibration_cells()
    all_validation_low = all(_ok_below(cell, low) for cell in validation)
    all_calibration_high = all(_ok_at_least(cell, high) for cell in calibration)
    if all_validation_low and all_calibration_high:
        return "過適合が主な結果"
    if all_validation_low:
        return "中止"
    return "判定保留"
