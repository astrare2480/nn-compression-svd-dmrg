r"""2-site の局所更新（結合・局所最適化・SVD 分割）。

参照元:
    notebooks/40_dmrg/00_fundamentals/00_two_site_dmrg_local_update_exercises.ipynb
    notebooks/40_dmrg/00_fundamentals/02_environment_two_site_dmrg_sweep_exercises.ipynb

隣接 2 site ``(n, n+1)`` の core を 4 階 tensor ``Theta`` へ結合し、有効 Hamiltonian の
最小固有ベクトルへ置き換え、SVD で 2 つの core へ分割し直す。分割では bond 次元の上限
``max_bond`` を適用でき、bond 次元が増減する（1-site 更新との違い）。

局所 solver は ``torch.linalg.eigh``（dense 対角化）で、Lanczos / Davidson は使わない。

autograd: 結合・有効 Hamiltonian・SVD 分割は ``detach()`` / ``no_grad()`` を使わず graph を
維持する。ただし ``eigh`` / ``svd`` は縮退した（特異）値で勾配が不安定になり得るため、
更新全体を end-to-end で微分可能な API とは位置づけない。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .effective_hamiltonian import _lowest_eigenpair, build_two_site_effective_hamiltonian
from .validation import (
    _require_finite,
    _validate_direction,
    _validate_environment,
    _validate_index,
    _validate_max_bond,
    _validate_mps_mpo_pair,
    _validate_same_dtype_device,
    _validate_tensor,
)


def merge_two_sites(left_core: torch.Tensor, right_core: torch.Tensor) -> torch.Tensor:
    r"""隣接 2 つの MPS core を 4 階 tensor ``(r_left, n_left, n_right, r_right)`` へ結合する。

    left_core: shape ``(r_left, n_left, r_mid)``。right_core: shape ``(r_mid, n_right, r_right)``。

    $$
    \Theta_{a s t c}=\sum_{b}A_{a s b}\,B_{b t c}
    $$

    dtype / device は一致している必要がある。入力は変更しない。autograd graph は維持する。
    """
    _validate_tensor(left_core, name="left_core", ndim=3)
    _validate_tensor(right_core, name="right_core", ndim=3)
    _validate_same_dtype_device(("left_core", left_core), ("right_core", right_core))
    if left_core.shape[2] != right_core.shape[0]:
        raise ValueError(
            f"left_core の right bond ({left_core.shape[2]}) と right_core の left bond "
            f"({right_core.shape[0]}) が一致しません。"
        )
    return torch.einsum("abc,cde->abde", left_core, right_core)


@dataclass(frozen=True)
class TwoSiteSplitResult:
    """2-site tensor の SVD 分割結果。

    left_core / right_core:
        分割後の core。形は ``(r_left, n_left, retained_rank)`` と
        ``(retained_rank, n_right, r_right)``。
    retained_rank:
        保持した bond 次元（Python の ``int``）。
    singular_values:
        切り捨て前の **全** 特異値（降順）。入力 tensor のスケールのまま（再正規化前）。
    discarded_squared_norm:
        捨てた特異値の二乗和 ``sum_{i > r} sigma_i^2``。
    discarded_weight:
        ``discarded_squared_norm / sum_i sigma_i^2``。入力のスケールに依存しない相対量。
    retained_norm:
        保持した特異値の二乗和の平方根 ``sqrt(sum_{i <= r} sigma_i^2)``。
    truncated_tensor_before_normalization:
        再正規化 **前** の、rank ``retained_rank`` へ切り捨てた 4 階 tensor
        （入力と同じ shape、ノルムは ``retained_norm``）。
    """

    left_core: torch.Tensor
    right_core: torch.Tensor
    retained_rank: int
    singular_values: torch.Tensor
    discarded_squared_norm: torch.Tensor
    discarded_weight: torch.Tensor
    retained_norm: torch.Tensor
    truncated_tensor_before_normalization: torch.Tensor


def split_two_site_tensor(
    two_site_tensor: torch.Tensor,
    *,
    direction: str,
    max_bond: int | None,
) -> TwoSiteSplitResult:
    r"""2-site tensor を SVD で 2 つの core へ分割する（bond 上限の適用と再正規化を含む）。

    two_site_tensor:
        shape ``(r_left, n_left, n_right, r_right)``。ノルムが 0 の tensor は ``ValueError``。
    direction:
        ``"left_to_right"``: 左 core が左正準（``U``）、特異値は右 core に置く（``sigma V^H``）。
        ``"right_to_left"``: 右 core が右正準（``V^H``）、特異値は左 core に置く（``U sigma``）。
    max_bond:
        保持する bond 次元の上限。``None`` なら切り捨てず、利用可能な rank
        ``min(r_left * n_left, n_right * r_right)`` をすべて保持する。1 以上の整数
        （bool 不可）。

    ノルムの規約:
        ``Theta`` を ``(r_left * n_left, n_right * r_right)`` 行列として reduced SVD し、
        上位 ``retained_rank`` 個を残す。残した特異値は **保持分のノルム ``retained_norm``
        で割って** 単位ノルムへ再正規化してから core へ置く。したがって

        - 分割後の 2 core を再結合した tensor のノルムは常に 1（入力のノルムによらない）、
        - ``max_bond=None`` で入力が単位ノルムなら、元の ``Theta`` を再現する、
        - 切り捨てがあると再結合した tensor は ``truncated_tensor_before_normalization /
          retained_norm`` に一致する。

        再正規化前の切り捨て tensor のノルムは ``retained_norm``、捨てた分は
        ``discarded_squared_norm``（``retained_norm**2 + discarded_squared_norm`` が切り捨て
        前の二乗ノルム）。``discarded_weight`` は両者の比で相対量。

        二乗和は、入力を最大絶対値で割った行列の SVD から相対的に計算する。float32 の
        ``1e-30`` / ``1e30`` や float64 の ``1e-200`` / ``1e200`` のように、特異値自体は
        有限でもその二乗が dtype の範囲を外れる入力を、零テンソルや非有限値と誤判定しない。
        ``discarded_weight`` と正規化後の core はこの相対量から作るので有限になる。
        ``discarded_squared_norm`` は絶対量なので、dtype に収まらないときは IEEE 丸めの
        ``inf``（overflow）または ``0``（underflow）になり、``NaN`` にはしない。weight の
        計算にはその絶対二乗和を使わない。core や ``retained_norm`` 自体が表現できない
        場合は ``ValueError``。

    入力は変更しない。autograd graph は維持する。
    """
    _validate_tensor(two_site_tensor, name="two_site_tensor", ndim=4)
    _validate_direction(direction)
    bond_limit = _validate_max_bond(max_bond)
    _require_finite([two_site_tensor], name="two_site_tensor")

    r_left, n_left, n_right, r_right = two_site_tensor.shape
    matrix = two_site_tensor.reshape(r_left * n_left, n_right * r_right)
    entry_scale = matrix.abs().amax()
    if not bool(torch.isfinite(entry_scale)):
        raise ValueError("two_site_tensor の最大絶対値が有限ではありません。")
    if bool(entry_scale == 0):
        raise ValueError("two_site_tensor のノルムの二乗が正の有限値ではありません（零 tensor など）。")

    # 最大絶対値で割ってから SVD する。特異値の二乗を入力 dtype のまま足すと、
    # 有限な 1e-30 / 1e30（float32）や 1e-200 / 1e200（float64）が 0 や inf になる。
    u, relative_singular_values, vh = torch.linalg.svd(matrix / entry_scale, full_matrices=False)
    if not bool(torch.isfinite(relative_singular_values).all()):
        raise ValueError(
            f"singular_values を dtype={two_site_tensor.dtype} で有限値として表現できません。"
        )
    singular_values = relative_singular_values * entry_scale
    relative_squared = relative_singular_values.square()
    total_relative_squared = relative_squared.sum()
    if not bool(total_relative_squared > 0) or not bool(torch.isfinite(total_relative_squared)):
        raise ValueError("two_site_tensor のノルムの二乗が正の有限値ではありません（零 tensor など）。")

    available_rank = relative_singular_values.shape[0]
    retained_rank = available_rank if bond_limit is None else min(available_rank, bond_limit)
    retained_relative_squared = relative_squared[:retained_rank].sum()
    discarded_relative_squared = relative_squared[retained_rank:].sum()
    retained_relative_norm = retained_relative_squared.sqrt()
    if not bool(retained_relative_norm > 0):
        raise ValueError("保持する特異値のノルムが 0 です。")
    # weight は相対二乗和の比。絶対二乗和（overflow / underflow し得る）では割らない。
    discarded_weight = discarded_relative_squared / total_relative_squared
    retained_norm = entry_scale * retained_relative_norm
    # 先にノルムへ戻してから二乗する。(scale**2) * 相対量 だと、両方が範囲外のとき NaN になる。
    discarded_squared_norm = (entry_scale * discarded_relative_squared.sqrt()).square()
    if bool(torch.isnan(discarded_squared_norm)):
        raise ValueError(
            f"discarded_squared_norm を dtype={two_site_tensor.dtype} で有限値として表現できません。"
        )

    u_kept = u[:, :retained_rank]
    vh_kept = vh[:retained_rank, :]
    normalized = relative_singular_values[:retained_rank] / retained_relative_norm
    if direction == "left_to_right":
        left_matrix = u_kept
        right_matrix = normalized.unsqueeze(1) * vh_kept
    else:
        left_matrix = u_kept * normalized.unsqueeze(0)
        right_matrix = vh_kept
    truncated = entry_scale * ((u_kept * relative_singular_values[:retained_rank].unsqueeze(0)) @ vh_kept)

    for name, tensor in (
        ("singular_values", singular_values),
        ("discarded_weight", discarded_weight),
        ("retained_norm", retained_norm),
        ("left_core", left_matrix),
        ("right_core", right_matrix),
        ("truncated_tensor_before_normalization", truncated),
    ):
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} を dtype={tensor.dtype} で有限値として表現できません。")
    return TwoSiteSplitResult(
        left_core=left_matrix.reshape(r_left, n_left, retained_rank),
        right_core=right_matrix.reshape(retained_rank, n_right, r_right),
        retained_rank=int(retained_rank),
        singular_values=singular_values,
        discarded_squared_norm=discarded_squared_norm,
        discarded_weight=discarded_weight,
        retained_norm=retained_norm,
        truncated_tensor_before_normalization=truncated.reshape(two_site_tensor.shape),
    )


@dataclass(frozen=True)
class TwoSiteUpdateResult:
    """1 回の 2-site 局所更新の結果。

    cores:
        ``pair_start`` と ``pair_start + 1`` の core だけを置き換えた新しい MPS core 列
        （他の core は複製）。
    pair_start:
        更新したペアの左 site の index。
    direction:
        SVD 分割で正準中心を動かす方向（``"left_to_right"`` / ``"right_to_left"``）。
    local_energy_before:
        更新前の 2-site tensor に対する局所 Rayleigh 商。
    local_minimum_energy:
        切り捨て **前** の局所最小固有値。切り捨て後に保存された MPS の全体 energy とは別の量。
    local_residual:
        ``|| H_eff v - E v ||``。局所固有値問題を解いた精度。
    split:
        SVD 分割の詳細（保持 rank、特異値、discarded weight など）。
    """

    cores: list[torch.Tensor]
    pair_start: int
    direction: str
    local_energy_before: torch.Tensor
    local_minimum_energy: torch.Tensor
    local_residual: torch.Tensor
    split: TwoSiteSplitResult


def _two_site_local_update(
    mps_cores,
    mpo_cores,
    pair_start: int,
    left_environment: torch.Tensor,
    right_environment: torch.Tensor,
    direction: str,
    max_bond: int | None,
) -> tuple[TwoSiteSplitResult, torch.Tensor, torch.Tensor, torch.Tensor]:
    """検査済みの入力に対する 2-site 局所更新の本体。core 列は作らず、分割結果と診断値だけ返す。"""
    theta_before = merge_two_sites(mps_cores[pair_start], mps_cores[pair_start + 1])
    effective_hamiltonian = build_two_site_effective_hamiltonian(
        left_environment, mpo_cores[pair_start], mpo_cores[pair_start + 1], right_environment
    )
    theta_vector = theta_before.reshape(-1)
    norm_squared = torch.dot(theta_vector, theta_vector)
    if not bool(norm_squared > 0):
        raise ValueError("更新前の 2-site tensor のノルムが 0 のため、局所 Rayleigh 商を定義できません。")
    local_energy_before = torch.dot(theta_vector, effective_hamiltonian @ theta_vector) / norm_squared

    local_minimum_energy, optimized_vector = _lowest_eigenpair(effective_hamiltonian)
    local_residual = torch.linalg.vector_norm(
        effective_hamiltonian @ optimized_vector - local_minimum_energy * optimized_vector
    )
    split = split_two_site_tensor(
        optimized_vector.reshape(theta_before.shape), direction=direction, max_bond=max_bond
    )
    return split, local_energy_before, local_minimum_energy, local_residual


def update_two_site_pair(
    mps_cores: list[torch.Tensor],
    mpo_cores: list[torch.Tensor],
    pair_start: int,
    left_environment: torch.Tensor,
    right_environment: torch.Tensor,
    *,
    direction: str,
    max_bond: int | None,
) -> TwoSiteUpdateResult:
    r"""隣接ペア ``(pair_start, pair_start + 1)`` に対し、1 回の 2-site 局所更新を行う。

    mps_cores / mpo_cores:
        MPS core 列 ``(r_left, n_k, r_right)`` と MPO core 列 ``(w_left, n_k, n_k, w_right)``。
    pair_start:
        更新するペアの左 site（``0 <= pair_start < d - 1``、bool 不可）。
    left_environment:
        site ``0..pair_start-1`` を縮約した ``(r, w, r)``
        （``build_left_environments(...)[pair_start]``）。
    right_environment:
        site ``pair_start+2..d-1`` を縮約した ``(r, w, r)``
        （``build_right_environments(...)[pair_start + 2]``）。
    direction, max_bond:
        ``split_two_site_tensor`` と同じ。

    処理:
        2 core を結合 -> 更新前の局所 Rayleigh 商を記録 -> 明示 ``H_eff`` の最小固有対を
        ``torch.linalg.eigh`` で求める -> 固有ベクトルを 4 階 tensor へ戻す ->
        ``split_two_site_tensor`` で分割し 2 core を置き換える。

    前提:
        局所問題が標準固有値問題になるのは、ペアより左が左正準、右が右正準の MPS から
        環境を作った場合である。この関数は正準性を検査しない。
        ``local_minimum_energy`` は切り捨て前の値で、``max_bond=None`` かつ環境が正準なら、
        更新後 MPS の全体 energy と一致する。

    入力は変更しない。autograd graph は維持するが、更新全体を end-to-end で微分可能な
    API とは位置づけない（``eigh`` / ``svd`` の縮退）。
    """
    _validate_mps_mpo_pair(mps_cores, mpo_cores)
    if len(mps_cores) < 2:
        raise ValueError("2-site 更新には 2 site 以上の MPS が必要です。")
    start = _validate_index(pair_start, len(mps_cores) - 1, name="pair_start")
    _validate_direction(direction)
    bond_limit = _validate_max_bond(max_bond)
    _validate_environment(
        left_environment,
        name="left_environment",
        mps_bond=mps_cores[start].shape[0],
        mpo_bond=mpo_cores[start].shape[0],
    )
    _validate_environment(
        right_environment,
        name="right_environment",
        mps_bond=mps_cores[start + 1].shape[2],
        mpo_bond=mpo_cores[start + 1].shape[3],
    )
    _validate_same_dtype_device(
        ("mps_cores[0]", mps_cores[0]),
        ("left_environment", left_environment),
        ("right_environment", right_environment),
    )

    split, before, minimum, residual = _two_site_local_update(
        mps_cores, mpo_cores, start, left_environment, right_environment, direction, bond_limit
    )
    updated_cores = [core.clone() for core in mps_cores]
    updated_cores[start] = split.left_core
    updated_cores[start + 1] = split.right_core
    return TwoSiteUpdateResult(
        cores=updated_cores,
        pair_start=start,
        direction=direction,
        local_energy_before=before,
        local_minimum_energy=minimum,
        local_residual=residual,
        split=split,
    )
