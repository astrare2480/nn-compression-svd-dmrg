"""DMRG テスト専用の helper（src の公開 API ではない）。

dense oracle（射影による有効 Hamiltonian、非 TFIM の MPO、dense 状態）は、src の実装を
再利用せず、この module 内で独立に組み立てる。
"""

from __future__ import annotations

import torch

from nn_compression.compression import tt_reconstruct
from nn_compression.dmrg import make_random_mps

DTYPES = (torch.float32, torch.float64)


def available_devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


def tolerances(dtype: torch.dtype) -> dict[str, float]:
    """``torch.testing.assert_close`` 用の許容誤差。"""
    if dtype == torch.float32:
        return {"atol": 2e-4, "rtol": 2e-4}
    return {"atol": 1e-10, "rtol": 1e-9}


def sweep_tolerances(dtype: torch.dtype) -> dict[str, float]:
    """sweep / DMRG の ``atol`` / ``rtol`` 引数。"""
    if dtype == torch.float32:
        return {"atol": 1e-3, "rtol": 1e-3}
    return {"atol": 1e-8, "rtol": 1e-8}


def random_mps(
    physical_dimensions,
    max_bond: int,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> list[torch.Tensor]:
    generator = torch.Generator(device=torch.device(device).type).manual_seed(seed)
    return make_random_mps(physical_dimensions, max_bond, generator=generator, dtype=dtype, device=device)


def dense_state(cores: list[torch.Tensor]) -> torch.Tensor:
    return tt_reconstruct(cores).reshape(-1)


def kron_all(matrices: list[torch.Tensor]) -> torch.Tensor:
    result = matrices[0]
    for matrix in matrices[1:]:
        result = torch.kron(result, matrix)
    return result


def symmetric_matrix(size: int, generator: torch.Generator, dtype, device) -> torch.Tensor:
    raw = torch.randn(size, size, generator=generator, dtype=torch.float64)
    return ((raw + raw.T) / 2).to(dtype=dtype, device=device)


def random_chain_hamiltonian(
    physical_dimensions,
    *,
    seed: int = 0,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """TFIM ではない小さな Hermitian chain Hamiltonian の ``(mpo_cores, dense_hamiltonian)``。

    ``H = sum_k F_k + sum_k B_k (x) A_{k+1}``（``F``, ``A``, ``B`` は site ごとの
    ランダムな実対称行列、物理次元は site ごとに任意）。MPO は bond 次元 3。
    dense 側は MPO を使わず、Kronecker 積で独立に作る。
    """
    generator = torch.Generator().manual_seed(seed)
    dims = list(physical_dimensions)
    num_sites = len(dims)
    fields = [symmetric_matrix(n, generator, dtype, device) for n in dims]
    ops_a = [symmetric_matrix(n, generator, dtype, device) for n in dims]
    ops_b = [symmetric_matrix(n, generator, dtype, device) for n in dims]
    identities = [torch.eye(n, dtype=dtype, device=device) for n in dims]
    zeros = [torch.zeros(n, n, dtype=dtype, device=device) for n in dims]

    def bulk(k: int) -> torch.Tensor:
        rows = [
            [identities[k], zeros[k], zeros[k]],
            [ops_a[k], zeros[k], zeros[k]],
            [fields[k], ops_b[k], identities[k]],
        ]
        return torch.stack([torch.stack(row, dim=-1) for row in rows], dim=0)

    cores = [bulk(k) for k in range(num_sites)]
    cores[0] = cores[0][2:3]  # 最下行
    cores[-1] = cores[-1][:, :, :, 0:1]  # 最左列

    dimension = 1
    for n in dims:
        dimension *= n
    dense = torch.zeros(dimension, dimension, dtype=dtype, device=device)
    for k in range(num_sites):
        factors = list(identities)
        factors[k] = fields[k]
        dense = dense + kron_all(factors)
    for k in range(num_sites - 1):
        factors = list(identities)
        factors[k] = ops_b[k]
        factors[k + 1] = ops_a[k + 1]
        dense = dense + kron_all(factors)
    return cores, dense


def _contract_chain(cores: list[torch.Tensor]) -> torch.Tensor:
    """core 列を縮約し ``(prod(n), right_bond)`` を返す（左端 bond は 1）。"""
    result = cores[0].reshape(cores[0].shape[1], cores[0].shape[2])
    for core in cores[1:]:
        result = torch.einsum("pa,asb->psb", result, core).reshape(-1, core.shape[2])
    return result


def _contract_right_chain(cores: list[torch.Tensor]) -> torch.Tensor:
    """core 列を縮約し ``(prod(n), left_bond)`` を返す（右端 bond は 1）。"""
    result = cores[0]
    for core in cores[1:]:
        merged = torch.einsum("bpx,xsy->bpsy", result, core)
        result = merged.reshape(merged.shape[0], -1, merged.shape[3])
    return result[:, :, 0].T


def projection_to_full_space(
    mps_cores: list[torch.Tensor],
    first_free: int,
    num_free: int,
) -> torch.Tensor:
    """局所自由度（site ``first_free`` から ``num_free`` 個）から全空間への等長写像 ``P``。

    ``P[(i_1..i_d), (a, s.., b)] = LB[(i_<), a] * delta(s, i_free) * RB[(i_>), b]``。
    ``LB`` / ``RB`` は左右の core を縮約した block。``H_eff = P^T H P``。
    """
    left_cores = mps_cores[:first_free]
    right_cores = mps_cores[first_free + num_free :]
    dtype, device = mps_cores[0].dtype, mps_cores[0].device
    if left_cores:
        left_block = _contract_chain(left_cores)
    else:
        left_block = torch.ones(1, 1, dtype=dtype, device=device)
    if right_cores:
        right_block = _contract_right_chain(right_cores)
    else:
        right_block = torch.ones(1, 1, dtype=dtype, device=device)
    left_bond = mps_cores[first_free].shape[0]
    right_bond = mps_cores[first_free + num_free - 1].shape[2]
    free_dims = [core.shape[1] for core in mps_cores[first_free : first_free + num_free]]
    free_identity = torch.eye(1, dtype=dtype, device=device)
    for n in free_dims:
        free_identity = torch.kron(free_identity, torch.eye(n, dtype=dtype, device=device))
    free_dimension = free_identity.shape[0]
    assert left_block.shape[1] == left_bond and right_block.shape[1] == right_bond
    projection = torch.einsum("la,sf,rb->lsrafb", left_block, free_identity, right_block)
    rows = left_block.shape[0] * free_dimension * right_block.shape[0]
    columns = left_bond * free_dimension * right_bond
    # 行 index は (l, s, r) の順、全空間の index (i_<, i_free, i_>) と一致する。
    return projection.reshape(rows, columns)
