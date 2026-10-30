r"""2-site DMRG の 1 方向 sweep。

参照元:
    notebooks/40_dmrg/00_fundamentals/02_environment_two_site_dmrg_sweep_exercises.ipynb
    （``sweep_left_to_right`` / ``sweep_right_to_left`` と ``SweepStepLog`` / ``SweepPassResult``）

Notebook では左向き・右向きの 2 関数だったが、環境の持ち方と走査順が違うだけなので、
公開 API は ``two_site_sweep`` 1 つにし、方向ごとの処理は private helper
``_directional_sweep`` に集約した。

正準形の扱い:
    ``two_site_sweep`` は入力 MPS を **複製して** 方向に応じた mixed-canonical form へ
    ``tt_canonicalize`` で変換してから走査する（``"left_to_right"`` は center=0、
    ``"right_to_left"`` は center=d-1）。QR による gauge 変換で状態は変わらない。
    走査中は、更新済みの外側が左（右）正準に保たれる。

    - ``"left_to_right"``: 終了時、site ``0..d-2`` が左正準で、正準中心は site ``d-1``。
    - ``"right_to_left"``: 終了時、site ``1..d-1`` が右正準で、正準中心は site ``0``。

autograd: 高レベルの探索処理であり、end-to-end で微分可能な API とは位置づけない
（``eigh`` / ``svd`` の縮退、``.item()`` による検査、core の反復的な置き換え）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..compression import tt_canonicalize
from .environment import _boundary_environment, _build_left, _build_right, _energy, _update_left, _update_right
from .two_site import _two_site_local_update
from .validation import _validate_direction, _validate_max_bond, _validate_mps_mpo_pair, _validate_tolerance


@dataclass(frozen=True)
class SweepStepLog:
    """sweep 中の 1 回の局所更新で記録する診断値（すべて Tensor / int）。

    direction:
        走査方向（``"left_to_right"`` / ``"right_to_left"``）。
    pair_start:
        更新したペアの左 site の index。
    retained_rank:
        SVD 後に保持した bond 次元。
    local_minimum_energy:
        切り捨て **前** の局所最小固有値。
    local_residual:
        局所固有値問題の残差 ``|| H_eff v - E v ||``。
    discarded_weight:
        SVD で捨てた状態成分の相対量（``TwoSiteSplitResult.discarded_weight``）。
    energy_after:
        切り捨て・再正規化後に **実際に保存された** MPS の全体 energy
        （MPS-MPO 縮約による Rayleigh 商）。
    """

    direction: str
    pair_start: int
    retained_rank: int
    local_minimum_energy: torch.Tensor
    local_residual: torch.Tensor
    discarded_weight: torch.Tensor
    energy_after: torch.Tensor


@dataclass(frozen=True)
class SweepPassResult:
    """1 方向の sweep 後の MPS と、更新ごとのログ（実行順）。"""

    cores: list[torch.Tensor]
    logs: list[SweepStepLog]


def _check_local_solution(
    local_energy_before: torch.Tensor,
    local_minimum_energy: torch.Tensor,
    local_residual: torch.Tensor,
    *,
    pair_start: int,
    atol: float,
    rtol: float,
) -> None:
    """局所固有値問題が許容内で解けているか確かめる（失敗は ``RuntimeError``）。

    - 残差: ``residual <= atol + rtol * |E_min|``。
    - 局所 energy が増えていないこと: ``E_min <= E_before + atol + rtol * |E_before|``
      （最小固有値は任意のベクトルの Rayleigh 商以下であるため、理論上は常に成立する。
      破れる場合は NaN などの数値失敗か環境の不整合を示す）。
    """
    residual_bound = atol + rtol * abs(float(local_minimum_energy))
    if not float(local_residual) <= residual_bound:
        raise RuntimeError(
            f"ペア ({pair_start}, {pair_start + 1}) の局所残差 {float(local_residual):.3e} が許容 "
            f"{residual_bound:.3e} を超えました。"
        )
    before = float(local_energy_before)
    energy_bound = before + atol + rtol * abs(before)
    if not float(local_minimum_energy) <= energy_bound:
        raise RuntimeError(
            f"ペア ({pair_start}, {pair_start + 1}) の局所最小固有値 {float(local_minimum_energy):.10g} が "
            f"更新前の局所 energy {before:.10g} を許容を超えて上回りました。"
        )


def _directional_sweep(
    cores: list[torch.Tensor],
    mpo_cores: list[torch.Tensor],
    direction: str,
    max_bond: int | None,
    atol: float,
    rtol: float,
) -> SweepPassResult:
    """検査・正準化済みの ``cores``（この関数が所有する list）を 1 方向に走査する。

    ``cores`` はその場で更新する。呼び出し側が入力を複製してから渡すこと。

    - ``"left_to_right"``: 最初に right environment を全 site 分作り、ペアを左端から右へ
      進める。更新後の左 core を left environment へ取り込む。
    - ``"right_to_left"``: 最初に left environment を全 site 分作り、ペアを右端から左へ
      進める。更新後の右 core を right environment へ取り込む。
    """
    num_sites = len(cores)
    left_to_right = direction == "left_to_right"
    logs: list[SweepStepLog] = []

    if left_to_right:
        right_environments = _build_right(cores, mpo_cores)
        left_environment = _boundary_environment(cores[0])
        pair_order = range(num_sites - 1)
    else:
        left_environments = _build_left(cores, mpo_cores)
        right_environment = _boundary_environment(cores[-1])
        pair_order = range(num_sites - 2, -1, -1)

    for pair_start in pair_order:
        if left_to_right:
            left_for_pair = left_environment
            right_for_pair = right_environments[pair_start + 2]
        else:
            left_for_pair = left_environments[pair_start]
            right_for_pair = right_environment

        split, before, minimum, residual = _two_site_local_update(
            cores, mpo_cores, pair_start, left_for_pair, right_for_pair, direction, max_bond
        )
        _check_local_solution(before, minimum, residual, pair_start=pair_start, atol=atol, rtol=rtol)
        cores[pair_start] = split.left_core
        cores[pair_start + 1] = split.right_core

        # 次のペアへ進むとき、中心から外れる側の core を environment へ取り込む。
        if left_to_right and pair_start < num_sites - 2:
            left_environment = _update_left(left_environment, cores[pair_start], mpo_cores[pair_start])
        if not left_to_right and pair_start > 0:
            right_environment = _update_right(
                right_environment, cores[pair_start + 1], mpo_cores[pair_start + 1]
            )

        logs.append(
            SweepStepLog(
                direction=direction,
                pair_start=pair_start,
                retained_rank=split.retained_rank,
                local_minimum_energy=minimum,
                local_residual=residual,
                discarded_weight=split.discarded_weight,
                energy_after=_energy(cores, mpo_cores),
            )
        )

    return SweepPassResult(cores=cores, logs=logs)


def two_site_sweep(
    mps_cores: list[torch.Tensor],
    mpo_cores: list[torch.Tensor],
    *,
    direction: str,
    max_bond: int | None,
    atol: float,
    rtol: float,
) -> SweepPassResult:
    r"""2-site DMRG の sweep を 1 方向に 1 回行う。

    mps_cores / mpo_cores:
        MPS core 列と MPO core 列（2 site 以上）。物理次元は site ごとに任意（2 でなくてよい）。
        MPS は正規化・正準化されている必要はない。
    direction:
        ``"left_to_right"``（ペアを左端 ``(0, 1)`` から右端 ``(d-2, d-1)`` へ）または
        ``"right_to_left"``（右端から左端へ）。
    max_bond:
        各局所 SVD で保持する bond 次元の上限（``None`` は切り捨てなし）。
    atol, rtol:
        局所固有値問題の健全性検査の許容誤差（有限・非負）。各更新で
        ``残差 <= atol + rtol * |E_min|``、および ``E_min <= E_before + atol + rtol * |E_before|``
        を確かめ、破れたら ``RuntimeError``。MPS-MPO の energy や環境の一致を判定する
        値ではない。``float32`` では ``float64`` より大きい値が必要になる。

    戻り値:
        ``SweepPassResult``。``cores`` は新しい MPS core 列（入力とは独立）、``logs`` は
        ``d - 1`` 個の ``SweepStepLog``（実行順）。``energy_after`` は各更新後の全体 energy で、
        切り捨てがあれば単調減少するとは限らない。

    正準形:
        入力を複製して方向に応じた mixed-canonical form へ ``tt_canonicalize`` したうえで
        走査する。終了時の正準形は module docstring を参照。
        environment は走査の進行に合わせて正しい順序で更新される。

    入力は変更しない。高レベルの探索処理であり、end-to-end で微分可能な API ではない。
    """
    _validate_mps_mpo_pair(mps_cores, mpo_cores)
    if len(mps_cores) < 2:
        raise ValueError("2-site sweep には 2 site 以上の MPS が必要です。")
    _validate_direction(direction)
    bond_limit = _validate_max_bond(max_bond)
    absolute = _validate_tolerance(atol, name="atol")
    relative = _validate_tolerance(rtol, name="rtol")

    center = 0 if direction == "left_to_right" else len(mps_cores) - 1
    cores = tt_canonicalize(list(mps_cores), center=center)
    return _directional_sweep(cores, list(mpo_cores), direction, bond_limit, absolute, relative)
