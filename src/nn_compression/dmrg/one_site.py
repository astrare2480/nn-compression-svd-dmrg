r"""1-site の局所更新（中心 core を有効 Hamiltonian の最小固有ベクトルへ置き換える）。

参照元:
    notebooks/40_dmrg/00_fundamentals/01_mpo_environment_one_site_local_update_exercises.ipynb
    （最後のセルに直接書かれていた「中心 core -> H_eff -> 最小固有対 -> 中心 core 更新」）

この module は **1 回の局所更新** だけを提供する。1-site の sweep（中心の移動・
environment の更新・bond 次元の扱い）は実装しない。1-site 更新は bond 次元を増やせない
ため、bond 次元の変更が必要なら 2-site 更新（``update_two_site_pair``）を使う。

局所問題が標準固有値問題になるのは、``left_environment`` / ``right_environment`` が
中心以外を正準形にした MPS から作られているときである。この module は正準性を検査せず、
呼び出し側（sweep など）の責務とする。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .effective_hamiltonian import _lowest_eigenpair, build_one_site_effective_hamiltonian
from .validation import (
    _validate_environment,
    _validate_index,
    _validate_mps_mpo_pair,
    _validate_same_dtype_device,
)


@dataclass(frozen=True)
class OneSiteUpdateResult:
    """1 回の 1-site 局所更新の結果。

    cores:
        中心 core だけを置き換えた新しい MPS core 列（他の core は複製）。
    center_site:
        更新した site の index。
    local_energy_before:
        更新前の中心 core に対する局所 Rayleigh 商 ``c^T H_eff c / c^T c``。
    local_minimum_energy:
        ``H_eff`` の最小固有値（更新後の局所 energy）。
    local_residual:
        ``|| H_eff v - E v ||``（``v`` は正規化された最小固有ベクトル）。局所固有値問題を
        解いた精度で、全体 Hamiltonian に対する残差ではない。
    """

    cores: list[torch.Tensor]
    center_site: int
    local_energy_before: torch.Tensor
    local_minimum_energy: torch.Tensor
    local_residual: torch.Tensor


def update_one_site_core(
    mps_cores: list[torch.Tensor],
    mpo_cores: list[torch.Tensor],
    center_site: int,
    left_environment: torch.Tensor,
    right_environment: torch.Tensor,
) -> OneSiteUpdateResult:
    r"""中心 ``center_site`` の core を、1-site 有効 Hamiltonian の最小固有ベクトルで置き換える。

    mps_cores / mpo_cores:
        MPS core 列 ``(r_left, n_k, r_right)`` と MPO core 列 ``(w_left, n_k, n_k, w_right)``。
    center_site:
        更新する site（``0 <= center_site < d``、bool 不可）。
    left_environment:
        site ``0..center_site-1`` を縮約した ``(r, w, r)``（``build_left_environments(...)[center_site]``）。
    right_environment:
        site ``center_site+1..d-1`` を縮約した ``(r, w, r)``（``build_right_environments(...)[center_site + 1]``）。

    処理:
        現在の中心 core から更新前の局所 Rayleigh 商を求め、明示的な ``H_eff`` を作って
        ``torch.linalg.eigh`` で最小固有対を求め、固有ベクトルを中心 core の shape へ戻す。
        固有ベクトルは単位ノルムなので、環境が正準形なら更新後の MPS もノルム 1 になる。

    入力は変更しない。autograd graph は維持するが、``eigh`` を使うため縮退した固有値では
    勾配が不安定になり得る。
    """
    _validate_mps_mpo_pair(mps_cores, mpo_cores)
    site = _validate_index(center_site, len(mps_cores), name="center_site")
    center = mps_cores[site]
    _validate_environment(
        left_environment, name="left_environment", mps_bond=center.shape[0], mpo_bond=mpo_cores[site].shape[0]
    )
    _validate_environment(
        right_environment, name="right_environment", mps_bond=center.shape[2], mpo_bond=mpo_cores[site].shape[3]
    )
    _validate_same_dtype_device(
        ("mps_cores[center_site]", center),
        ("left_environment", left_environment),
        ("right_environment", right_environment),
    )

    effective_hamiltonian = build_one_site_effective_hamiltonian(
        left_environment, mpo_cores[site], right_environment
    )
    center_vector = center.reshape(-1)
    center_norm_squared = torch.dot(center_vector, center_vector)
    if not bool(center_norm_squared > 0):
        raise ValueError("中心 core のノルムが 0 のため、局所 Rayleigh 商を定義できません。")
    local_energy_before = torch.dot(center_vector, effective_hamiltonian @ center_vector) / center_norm_squared

    local_minimum_energy, optimized_vector = _lowest_eigenpair(effective_hamiltonian)
    local_residual = torch.linalg.vector_norm(
        effective_hamiltonian @ optimized_vector - local_minimum_energy * optimized_vector
    )

    updated_cores = [core.clone() for core in mps_cores]
    updated_cores[site] = optimized_vector.reshape(center.shape)
    return OneSiteUpdateResult(
        cores=updated_cores,
        center_site=site,
        local_energy_before=local_energy_before,
        local_minimum_energy=local_minimum_energy,
        local_residual=local_residual,
    )
