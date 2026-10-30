"""``dmrg.one_site``（1-site 局所更新）のテスト。"""

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
    sweep_tolerances,
    tolerances,
)
from nn_compression.compression import tt_canonicalize
from nn_compression.dmrg import (
    OneSiteUpdateResult,
    build_left_environments,
    build_right_environments,
    mps_mpo_energy,
    update_one_site_core,
)

CASES = {
    "d2_rank2": ([2, 2, 2, 2], 2),
    "d3_rank3": ([3, 3, 3, 3], 3),
    "d5_rank4": ([5, 4, 5], 4),
    "mixed_rank1": ([2, 3, 4, 2, 3], 1),
    "mixed_rank3": ([2, 3, 4, 2, 3], 3),
}


def _canonical_setup(dims, max_bond, center, *, dtype=torch.float64, device="cpu"):
    mpo, dense = random_chain_hamiltonian(dims, seed=3, dtype=dtype, device=device)
    mps = random_mps(dims, max_bond, seed=4, dtype=dtype, device=device)
    mps = tt_canonicalize(mps, center)
    left = build_left_environments(mps, mpo)
    right = build_right_environments(mps, mpo)
    return mps, mpo, dense, left, right


@pytest.mark.parametrize("dims,max_bond", CASES.values(), ids=CASES.keys())
def test_update_matches_dense_projection_ground_state(dims, max_bond):
    """(5)(6)(16)(17)(18) 局所最小固有値は P^T H P の最小固有値、残差は十分小さい。"""
    for site in range(len(dims)):
        mps, mpo, dense, left, right = _canonical_setup(dims, max_bond, site)
        result = update_one_site_core(mps, mpo, site, left[site], right[site + 1])
        assert isinstance(result, OneSiteUpdateResult)
        assert result.center_site == site

        projection = projection_to_full_space(mps, site, 1)
        expected = torch.linalg.eigvalsh(projection.T @ dense @ projection)[0]
        torch.testing.assert_close(result.local_minimum_energy, expected, **tolerances(torch.float64))
        assert result.local_residual.item() <= 1e-10
        assert result.local_minimum_energy.item() <= result.local_energy_before.item() + 1e-12

        assert result.cores[site].shape == mps[site].shape
        norm = torch.linalg.vector_norm(result.cores[site])
        torch.testing.assert_close(norm, torch.tensor(1.0, dtype=torch.float64))


@pytest.mark.parametrize("dims,max_bond", CASES.values(), ids=CASES.keys())
def test_canonical_environment_makes_local_energy_equal_global_energy(dims, max_bond):
    """中心以外が正準なら、局所 Rayleigh 商は全体 energy と一致し、更新後の全体 energy は局所最小値になる。"""
    site = len(dims) // 2
    mps, mpo, _, left, right = _canonical_setup(dims, max_bond, site)
    result = update_one_site_core(mps, mpo, site, left[site], right[site + 1])
    torch.testing.assert_close(
        result.local_energy_before, mps_mpo_energy(mps, mpo), **tolerances(torch.float64)
    )
    torch.testing.assert_close(
        mps_mpo_energy(result.cores, mpo), result.local_minimum_energy, **tolerances(torch.float64)
    )


def test_only_the_center_core_changes_and_inputs_are_not_mutated():
    """(22) 中心以外は値が同じ別 Tensor。入力は変更されない。"""
    dims = [2, 3, 2, 3]
    site = 1
    mps, mpo, _, left, right = _canonical_setup(dims, 2, site)
    operands = [*mps, *mpo, left[site], right[site + 1]]
    before = copy.deepcopy(operands)
    result = update_one_site_core(mps, mpo, site, left[site], right[site + 1])
    for a, b in zip(operands, before):
        assert torch.equal(a, b)
    for index, (new, old) in enumerate(zip(result.cores, mps)):
        if index != site:
            assert torch.equal(new, old)
            assert new.data_ptr() != old.data_ptr()
    assert not torch.allclose(result.cores[site], mps[site])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", available_devices(), ids=str)
def test_dtype_and_device_preserved(dtype, device):
    """(19)(20)(21)"""
    site = 1
    mps, mpo, dense, left, right = _canonical_setup([2, 3, 2, 3], 2, site, dtype=dtype, device=device)
    result = update_one_site_core(mps, mpo, site, left[site], right[site + 1])
    for tensor in (*result.cores, result.local_energy_before, result.local_minimum_energy, result.local_residual):
        assert tensor.dtype == dtype and tensor.device.type == device.type
    projection = projection_to_full_space(mps, site, 1)
    expected = torch.linalg.eigvalsh(projection.T @ dense @ projection)[0]
    torch.testing.assert_close(result.local_minimum_energy, expected, **tolerances(dtype))
    assert result.local_residual.item() <= sweep_tolerances(dtype)["atol"]


def test_invalid_arguments_are_rejected():
    """(23)(24)"""
    mps, mpo, _, left, right = _canonical_setup([2, 3, 2, 3], 2, 1)
    with pytest.raises(TypeError):
        update_one_site_core(mps, mpo, True, left[1], right[2])
    with pytest.raises(TypeError):
        update_one_site_core(mps, mpo, 1.0, left[1], right[2])
    for bad in (-1, 4):
        with pytest.raises(ValueError, match="center_site"):
            update_one_site_core(mps, mpo, bad, left[1], right[2])
    with pytest.raises(ValueError, match="shape"):
        update_one_site_core(mps, mpo, 1, left[0], right[2])
    with pytest.raises(ValueError, match="shape"):
        update_one_site_core(mps, mpo, 1, left[1], right[4])
    with pytest.raises(TypeError, match="dtype"):
        update_one_site_core(mps, mpo, 1, left[1].float(), right[2])
    with pytest.raises(ValueError, match="device"):
        update_one_site_core(mps, mpo, 1, left[1], right[2].to("meta"))
    with pytest.raises(ValueError, match="site 数"):
        update_one_site_core(mps[:-1], mpo, 1, left[1], right[2])
    with pytest.raises(TypeError):
        update_one_site_core([core.to(torch.complex128) for core in mps], mpo, 1, left[1], right[2])


def test_zero_center_core_is_rejected():
    mps, mpo, _, left, right = _canonical_setup([2, 3, 2, 3], 2, 1)
    zero_center = list(mps)
    zero_center[1] = torch.zeros_like(mps[1])
    with pytest.raises(ValueError, match="ノルムが 0"):
        update_one_site_core(zero_center, mpo, 1, left[1], right[2])


def test_non_hermitian_mpo_is_rejected_instead_of_silently_symmetrized():
    mps, mpo, _, left, right = _canonical_setup([2, 3, 2, 3], 2, 1)
    skewed = [core.clone() for core in mpo]
    skewed[1] = skewed[1] + 0.5 * torch.randn_like(skewed[1])
    left = build_left_environments(mps, skewed)
    right = build_right_environments(mps, skewed)
    with pytest.raises(ValueError, match="対称"):
        update_one_site_core(mps, skewed, 1, left[1], right[2])


def test_update_backward_is_finite_for_non_degenerate_problem():
    """(26) 局所 energy の勾配が有限（縮退しない問題に限る）。"""
    site = 1
    mps, mpo, _, left, right = _canonical_setup([2, 3, 2, 3], 2, site)
    environment = left[site].clone().requires_grad_(True)
    result = update_one_site_core(mps, mpo, site, environment, right[site + 1])
    result.local_minimum_energy.backward()
    assert environment.grad is not None and torch.isfinite(environment.grad).all()
