"""``dmrg.effective_hamiltonian``（明示行列と matrix-free 作用）のテスト。"""

from __future__ import annotations

import copy

import pytest
import torch

from _dmrg_testing import (
    DTYPES,
    available_devices,
    projection_to_full_space,
    random_chain_hamiltonian,
    random_mps,
    tolerances,
)
from nn_compression.dmrg import (
    apply_one_site_effective_hamiltonian,
    apply_two_site_effective_hamiltonian,
    build_left_environments,
    build_one_site_effective_hamiltonian,
    build_right_environments,
    build_two_site_effective_hamiltonian,
    merge_two_sites,
)

CASES = {
    "d2": ([2, 2, 2, 2], 3),
    "d3": ([3, 3, 3, 3], 3),
    "d5": ([5, 4, 5], 4),
    "mixed_rank1": ([2, 3, 4, 2, 3], 1),
    "mixed_rank2": ([2, 3, 4, 2, 3], 2),
}


def _setup(dims, max_bond, *, dtype=torch.float64, device="cpu", seed=0):
    mpo, dense = random_chain_hamiltonian(dims, seed=seed, dtype=dtype, device=device)
    mps = random_mps(dims, max_bond, seed=seed + 1, dtype=dtype, device=device)
    return mps, mpo, dense, build_left_environments(mps, mpo), build_right_environments(mps, mpo)


@pytest.mark.parametrize("dims,max_bond", CASES.values(), ids=CASES.keys())
def test_one_site_explicit_matrix_free_and_dense_projection_agree(dims, max_bond):
    """(5)(16)(17)(18) 全 site で 明示 H_eff = matrix-free = P^T H P。"""
    mps, mpo, dense, left, right = _setup(dims, max_bond)
    generator = torch.Generator().manual_seed(7)
    for site in range(len(dims)):
        explicit = build_one_site_effective_hamiltonian(left[site], mpo[site], right[site + 1])
        projection = projection_to_full_space(mps, site, 1)
        torch.testing.assert_close(explicit, projection.T @ dense @ projection, **tolerances(torch.float64))
        torch.testing.assert_close(explicit, explicit.T, **tolerances(torch.float64))

        vector = torch.randn(mps[site].shape, generator=generator, dtype=torch.float64)
        applied = apply_one_site_effective_hamiltonian(left[site], mpo[site], right[site + 1], vector)
        assert applied.shape == vector.shape
        torch.testing.assert_close(
            applied.reshape(-1), explicit @ vector.reshape(-1), **tolerances(torch.float64)
        )


@pytest.mark.parametrize("dims,max_bond", CASES.values(), ids=CASES.keys())
def test_two_site_explicit_matrix_free_and_dense_projection_agree(dims, max_bond):
    """(7)(16)(17)(18) 全ペアで 明示 H_eff = matrix-free = P^T H P。"""
    mps, mpo, dense, left, right = _setup(dims, max_bond)
    generator = torch.Generator().manual_seed(8)
    for pair in range(len(dims) - 1):
        explicit = build_two_site_effective_hamiltonian(
            left[pair], mpo[pair], mpo[pair + 1], right[pair + 2]
        )
        projection = projection_to_full_space(mps, pair, 2)
        torch.testing.assert_close(explicit, projection.T @ dense @ projection, **tolerances(torch.float64))
        torch.testing.assert_close(explicit, explicit.T, **tolerances(torch.float64))

        theta = merge_two_sites(mps[pair], mps[pair + 1])
        assert explicit.shape == (theta.numel(), theta.numel())
        vector = torch.randn(theta.shape, generator=generator, dtype=torch.float64)
        applied = apply_two_site_effective_hamiltonian(
            left[pair], mpo[pair], mpo[pair + 1], right[pair + 2], vector
        )
        assert applied.shape == vector.shape
        torch.testing.assert_close(
            applied.reshape(-1), explicit @ vector.reshape(-1), **tolerances(torch.float64)
        )
        # Rayleigh 商が全空間のそれと一致する（局所ベクトルから全状態を作る）
        full = projection @ theta.reshape(-1)
        local_value = theta.reshape(-1) @ explicit @ theta.reshape(-1)
        torch.testing.assert_close(local_value, full @ dense @ full, **tolerances(torch.float64))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", available_devices(), ids=str)
def test_dtype_device_preserved(dtype, device):
    """(19)(20)(21)"""
    mps, mpo, dense, left, right = _setup([2, 3, 2, 3], 2, dtype=dtype, device=device)
    explicit = build_one_site_effective_hamiltonian(left[1], mpo[1], right[2])
    assert explicit.dtype == dtype and explicit.device.type == device.type
    projection = projection_to_full_space(mps, 1, 1)
    torch.testing.assert_close(explicit, projection.T @ dense @ projection, **tolerances(dtype))
    vector = torch.ones(mps[1].shape, dtype=dtype, device=device)
    applied = apply_one_site_effective_hamiltonian(left[1], mpo[1], right[2], vector)
    assert applied.dtype == dtype and applied.device.type == device.type
    torch.testing.assert_close(applied.reshape(-1), explicit @ vector.reshape(-1), **tolerances(dtype))

    explicit2 = build_two_site_effective_hamiltonian(left[1], mpo[1], mpo[2], right[3])
    theta = merge_two_sites(mps[1], mps[2])
    applied2 = apply_two_site_effective_hamiltonian(left[1], mpo[1], mpo[2], right[3], theta)
    assert explicit2.dtype == dtype and applied2.dtype == dtype
    torch.testing.assert_close(applied2.reshape(-1), explicit2 @ theta.reshape(-1), **tolerances(dtype))


def test_effective_hamiltonian_inputs_are_not_mutated():
    """(22)"""
    mps, mpo, _, left, right = _setup([2, 3, 2, 3], 2)
    operands = [left[1], mpo[1], mpo[2], right[3]]
    theta = merge_two_sites(mps[1], mps[2])
    before = copy.deepcopy(operands + [theta])
    build_two_site_effective_hamiltonian(*operands)
    apply_two_site_effective_hamiltonian(*operands, theta)
    build_one_site_effective_hamiltonian(left[1], mpo[1], right[2])
    apply_one_site_effective_hamiltonian(left[1], mpo[1], right[2], mps[1])
    for a, b in zip(operands + [theta], before):
        assert torch.equal(a, b)


def test_backward_through_effective_hamiltonian_is_finite():
    """(26)"""
    mps, mpo, _, left, right = _setup([2, 3, 2, 3], 2)
    left_env = left[1].clone().requires_grad_(True)
    right_env = right[3].clone().requires_grad_(True)
    theta = merge_two_sites(mps[1], mps[2]).clone().requires_grad_(True)
    applied = apply_two_site_effective_hamiltonian(left_env, mpo[1], mpo[2], right_env, theta)
    (applied * theta).sum().backward()
    for tensor in (left_env, right_env, theta):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()

    left_env = left[1].clone().requires_grad_(True)
    build_two_site_effective_hamiltonian(left_env, mpo[1], mpo[2], right[3]).sum().backward()
    assert torch.isfinite(left_env.grad).all()

    center = mps[1].clone().requires_grad_(True)
    out = apply_one_site_effective_hamiltonian(left[1], mpo[1], right[2], center)
    out.sum().backward()
    assert torch.isfinite(center.grad).all()


def test_invalid_operands_are_rejected():
    """(23)"""
    mps, mpo, _, left, right = _setup([2, 3, 2, 3], 2)
    theta = merge_two_sites(mps[1], mps[2])
    with pytest.raises(ValueError, match="shape"):
        build_one_site_effective_hamiltonian(left[0], mpo[1], right[2])  # 環境の取り違え
    with pytest.raises(ValueError, match="shape"):
        build_one_site_effective_hamiltonian(left[1], mpo[1], right[4])  # 境界 environment（bond 1）
    with pytest.raises(ValueError, match="物理次元"):
        build_one_site_effective_hamiltonian(left[1], mpo[1][:, :, :2], right[2])
    with pytest.raises(ValueError, match="bond"):
        build_two_site_effective_hamiltonian(left[1], mpo[1], mpo[2][:2], right[3])
    with pytest.raises(ValueError, match="shape"):
        build_two_site_effective_hamiltonian(left[1], mpo[1], mpo[2], right[4])
    with pytest.raises(ValueError, match="shape"):
        apply_one_site_effective_hamiltonian(left[1], mpo[1], right[2], mps[2])
    with pytest.raises(ValueError, match="shape"):
        apply_two_site_effective_hamiltonian(left[1], mpo[1], mpo[2], right[3], theta[..., :1])
    with pytest.raises(ValueError, match="4階"):
        apply_two_site_effective_hamiltonian(left[1], mpo[1], mpo[2], right[3], theta[0])
    with pytest.raises(TypeError, match="dtype"):
        build_two_site_effective_hamiltonian(left[1].float(), mpo[1], mpo[2], right[3])
    with pytest.raises(ValueError, match="device"):
        build_one_site_effective_hamiltonian(left[1].to("meta"), mpo[1], right[2])
    with pytest.raises(TypeError):
        build_one_site_effective_hamiltonian(left[1].to(torch.complex128), mpo[1], right[2])
    with pytest.raises(TypeError):
        build_one_site_effective_hamiltonian(left[1].to(torch.bfloat16), mpo[1], right[2])
    with pytest.raises(TypeError, match="Tensor"):
        build_one_site_effective_hamiltonian([1.0], mpo[1], right[2])
