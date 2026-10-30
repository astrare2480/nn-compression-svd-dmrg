r"""2-site DMRG の往復（round trip）と、energy 安定による停止判定。

参照元:
    notebooks/40_dmrg/00_fundamentals/03_dmrg_convergence_and_bond_dimension_exercises.ipynb
    （``run_one_round_trip`` / ``next_stability_count`` / ``choose_stop_reason`` /
    ``run_dmrg_until_energy_stable``）

Notebook 03 の反復処理は dense Hamiltonian、厳密基底 energy、全残差・energy 分散を引数に
とって診断値を記録していたが、それらは停止判定に使われていなかった。ここでは DMRG 本体を
それらから切り離し、停止判定は MPS-MPO 縮約で求めた energy の往復間変化だけで行う。
dense の残差・分散が必要なら ``compute_dense_residual_and_variance`` を呼び出し側で使う。

用語:
    round trip: ``"left_to_right"`` の sweep と、続く ``"right_to_left"`` の sweep を 1 組にした
    もの。終了時の正準中心は左端（site 0）に戻る。
    round_trip 0: 局所更新を行う前の初期状態。

停止規則（``k`` = 完了した往復数、``c_k`` = 安定の連続成立回数）:
    ``energy_change = |E_k - E_{k-1}|``（``E_0`` は初期状態の energy）が
    ``energy_change <= energy_tolerance``（境界を含む）なら ``c_k = c_{k-1} + 1``、そうでなければ
    ``c_k = 0``。``k >= min_round_trips`` かつ ``c_k >= required_consecutive`` なら
    ``"converged"``。そうでなく ``k >= max_round_trips`` なら ``"max_round_trips_reached"``。
    両方が同時に成立したときは ``"converged"`` を優先する。``"converged"`` は energy が設定した
    基準で安定したことを意味するだけで、基底状態・固有状態への到達は保証しない。

autograd: 高レベルの反復処理であり、end-to-end で微分可能な API ではない（``eigh`` / ``svd``
の縮退、``.item()`` による停止判定、core の反復的な置き換え）。``run_two_site_dmrg`` は
``torch.no_grad()`` の下で実行し、返す core は graph から切り離される。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from ..compression import tt_ranks
from .environment import _energy
from .sweep import SweepStepLog, two_site_sweep
from .validation import (
    _validate_max_bond,
    _validate_mps_mpo_pair,
    _validate_positive_int,
    _validate_tolerance,
)


@dataclass(frozen=True)
class DMRGRoundTripResult:
    """1 往復後の MPS と、往復中の全局所更新ログ（実行順、長さ ``2 * (num_sites - 1)``）。"""

    cores: list[torch.Tensor]
    logs: list[SweepStepLog]


@dataclass(frozen=True)
class DMRGRoundTripRecord:
    """初期状態（``round_trip == 0``）または 1 往復終了時の履歴 1 件。

    round_trip:
        完了した往復の数。初期状態は 0。
    energy:
        その時点の MPS の全体 energy（MPS-MPO 縮約による Rayleigh 商）。
    energy_change:
        直前の往復境界からの ``|energy 差|``。比較対象のない初期状態では ``None``
        （0 という測定値にはしない）。
    actual_bond_dimensions:
        その時点の内部 bond 次元 ``(r_1, ..., r_{d-1})``。
    max_discarded_weight:
        その往復の全局所 SVD の discarded weight の最大値。局所的な切り捨ての要約であり、
        最終状態の誤差や基底 energy の誤差ではない。初期状態では ``None``。
    stability_count:
        energy 安定条件が連続して成立した回数。
    """

    round_trip: int
    energy: float
    energy_change: float | None
    actual_bond_dimensions: tuple[int, ...]
    max_discarded_weight: float | None
    stability_count: int


@dataclass(frozen=True)
class DMRGResult:
    """停止した DMRG 反復の最終 MPS、履歴（初期状態を含む）、終了理由。

    stop_reason:
        ``"converged"``（energy 安定条件が成立）または ``"max_round_trips_reached"``。
    """

    cores: list[torch.Tensor]
    history: list[DMRGRoundTripRecord]
    stop_reason: str


def _next_stability_count(
    energy_change: float | None,
    *,
    tolerance: float,
    current_count: int,
) -> int:
    """今回の energy 変化を反映した、安定条件の連続成立回数を返す。

    ``energy_change is None``（比較対象なし）は 0 に戻す。非有限値は ``FloatingPointError``。
    ``energy_change <= tolerance``（境界を含む）なら ``current_count + 1``、超えたら 0。
    """
    if tolerance < 0.0:
        raise ValueError("tolerance は非負である必要があります。")
    if current_count < 0:
        raise ValueError("current_count は非負である必要があります。")
    if energy_change is None:
        return 0
    if not math.isfinite(energy_change):
        raise FloatingPointError(f"energy_change が有限値ではありません: {energy_change!r}")
    return current_count + 1 if energy_change <= tolerance else 0


def _choose_stop_reason(
    *,
    round_trip: int,
    stable_count: int,
    min_round_trips: int,
    max_round_trips: int,
    required_consecutive: int,
) -> str:
    """現在の履歴に対応する終了理由 ``"converged"`` / ``"max_round_trips_reached"``、または ``"continue"``。"""
    if not 1 <= min_round_trips <= max_round_trips:
        raise ValueError("1 <= min_round_trips <= max_round_trips である必要があります。")
    if required_consecutive < 1:
        raise ValueError("required_consecutive は 1 以上である必要があります。")
    if round_trip >= min_round_trips and stable_count >= required_consecutive:
        return "converged"
    if round_trip >= max_round_trips:
        return "max_round_trips_reached"
    return "continue"


def run_two_site_dmrg_round_trip(
    mps_cores: list[torch.Tensor],
    mpo_cores: list[torch.Tensor],
    *,
    max_bond: int | None,
    atol: float,
    rtol: float,
) -> DMRGRoundTripResult:
    """``"left_to_right"`` の sweep と ``"right_to_left"`` の sweep を 1 往復として実行する。

    左向きの結果の MPS を、そのまま右向き sweep へ渡す。戻り値の ``logs`` は両 sweep の
    更新を実行順に連結したもの（長さ ``2 * (num_sites - 1)``）で、終了時の正準中心は
    site 0。``max_bond`` / ``atol`` / ``rtol`` の意味は ``two_site_sweep`` に同じ。
    入力は変更しない。end-to-end で微分可能な API ではない。
    """
    forward = two_site_sweep(
        mps_cores, mpo_cores, direction="left_to_right", max_bond=max_bond, atol=atol, rtol=rtol
    )
    backward = two_site_sweep(
        forward.cores, mpo_cores, direction="right_to_left", max_bond=max_bond, atol=atol, rtol=rtol
    )
    return DMRGRoundTripResult(cores=backward.cores, logs=[*forward.logs, *backward.logs])


def run_two_site_dmrg(
    initial_cores: list[torch.Tensor],
    mpo_cores: list[torch.Tensor],
    *,
    max_bond: int | None,
    min_round_trips: int,
    max_round_trips: int,
    energy_tolerance: float,
    required_consecutive: int,
    atol: float,
    rtol: float,
) -> DMRGResult:
    """往復を反復し、energy 安定条件または往復上限で止まる 2-site DMRG を実行する。

    initial_cores / mpo_cores:
        初期 MPS（2 site 以上、正規化・正準化は不要）と Hamiltonian の MPO。dense
        Hamiltonian、厳密基底 energy、全状態ベクトルは使わない。
    max_bond:
        各局所 SVD の bond 次元の上限（``None`` は切り捨てなし）。
    min_round_trips, max_round_trips:
        ``1 <= min_round_trips <= max_round_trips`` の整数（bool 不可）。
    energy_tolerance:
        往復間の energy 変化の許容値（有限・非負）。``energy_change <= energy_tolerance`` を
        安定とみなす（境界を含む）。
    required_consecutive:
        停止に必要な、安定の連続成立回数（1 以上）。
    atol, rtol:
        各局所更新の健全性検査の許容誤差（``two_site_sweep`` を参照）。

    戻り値:
        ``DMRGResult``。``history`` は初期状態（``round_trip == 0``、``energy_change`` と
        ``max_discarded_weight`` は ``None``）から始まり、完了した往復ごとに 1 件。
        停止規則と ``stop_reason`` は module docstring を参照。

    入力は変更しない。``torch.no_grad()`` の下で実行し、返す core は autograd graph から
    切り離される（end-to-end で微分可能な API ではない）。energy や残差の単調性は要求しない。
    """
    _validate_mps_mpo_pair(initial_cores, mpo_cores)
    if len(initial_cores) < 2:
        raise ValueError("2-site DMRG には 2 site 以上の MPS が必要です。")
    bond_limit = _validate_max_bond(max_bond)
    minimum = _validate_positive_int(min_round_trips, name="min_round_trips")
    maximum = _validate_positive_int(max_round_trips, name="max_round_trips")
    if minimum > maximum:
        raise ValueError(
            f"min_round_trips={minimum} は max_round_trips={maximum} 以下である必要があります。"
        )
    tolerance = _validate_tolerance(energy_tolerance, name="energy_tolerance")
    required = _validate_positive_int(required_consecutive, name="required_consecutive")
    absolute = _validate_tolerance(atol, name="atol")
    relative = _validate_tolerance(rtol, name="rtol")

    with torch.no_grad():
        cores = [core.clone() for core in initial_cores]
        initial_energy = float(_energy(cores, mpo_cores))
        if not math.isfinite(initial_energy):
            raise FloatingPointError(f"初期状態の energy が有限値ではありません: {initial_energy!r}")
        history = [
            DMRGRoundTripRecord(
                round_trip=0,
                energy=initial_energy,
                energy_change=None,
                actual_bond_dimensions=tuple(tt_ranks(cores)[1:-1]),
                max_discarded_weight=None,
                stability_count=0,
            )
        ]
        previous_energy = initial_energy
        stable_count = 0

        for round_trip in range(1, maximum + 1):
            result = run_two_site_dmrg_round_trip(
                cores, mpo_cores, max_bond=bond_limit, atol=absolute, rtol=relative
            )
            cores = result.cores
            energy = float(result.logs[-1].energy_after)
            if not math.isfinite(energy):
                raise FloatingPointError(f"往復 {round_trip} 後の energy が有限値ではありません: {energy!r}")
            energy_change = abs(energy - previous_energy)
            stable_count = _next_stability_count(
                energy_change, tolerance=tolerance, current_count=stable_count
            )
            history.append(
                DMRGRoundTripRecord(
                    round_trip=round_trip,
                    energy=energy,
                    energy_change=energy_change,
                    actual_bond_dimensions=tuple(tt_ranks(cores)[1:-1]),
                    max_discarded_weight=max(float(log.discarded_weight) for log in result.logs),
                    stability_count=stable_count,
                )
            )
            previous_energy = energy

            stop_reason = _choose_stop_reason(
                round_trip=round_trip,
                stable_count=stable_count,
                min_round_trips=minimum,
                max_round_trips=maximum,
                required_consecutive=required,
            )
            if stop_reason != "continue":
                return DMRGResult(cores=cores, history=history, stop_reason=stop_reason)

    raise RuntimeError("最大往復数に達しても終了理由が決まりませんでした（停止規則の実装不整合）。")
