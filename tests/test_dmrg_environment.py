"""``dmrg.environment``（environment・期待値・energy・正規化）のテスト。"""

from __future__ import annotations

import copy

import pytest
import torch

from _dmrg_testing import (
    DTYPES,
    available_devices,
    dense_state,
    random_chain_hamiltonian,
    random_mps,
    sweep_tolerances,
    tolerances,
)
from nn_compression.compression import tt_fro_norm, tt_reconstruct
from nn_compression.dmrg import (
    build_left_environments,
    build_right_environments,
    build_tfim_dense_hamiltonian,
    build_tfim_mpo,
    mps_mpo_energy,
    mps_mpo_expectation,
    normalize_mps,
    run_two_site_dmrg,
    update_left_environment,
    update_right_environment,
)

PHYSICAL_CASES = {
    "d2": [2, 2, 2, 2],
    "d3": [3, 3, 3, 3],
    "d5": [5, 4, 5],
    "mixed": [2, 3, 4, 2, 3],
}


def _problem(dims, *, seed=0, dtype=torch.float64, device="cpu", max_bond=3):
    mpo, dense = random_chain_hamiltonian(dims, seed=seed, dtype=dtype, device=device)
    mps = random_mps(dims, max_bond, seed=seed + 1, dtype=dtype, device=device)
    return mps, mpo, dense


@pytest.mark.parametrize("dims", PHYSICAL_CASES.values(), ids=PHYSICAL_CASES.keys())
def test_environment_scalar_matches_dense_expectation(dims):
    """(1)(2)(16)(18) 左右 environment の末端 scalar が dense の <psi|H|psi> と一致する。"""
    mps, mpo, dense = _problem(dims)
    psi = dense_state(mps)
    expected = psi @ dense @ psi
    left = build_left_environments(mps, mpo)
    right = build_right_environments(mps, mpo)
    assert len(left) == len(right) == len(dims) + 1
    torch.testing.assert_close(left[-1].reshape(()), expected, **tolerances(torch.float64))
    torch.testing.assert_close(right[0].reshape(()), expected, **tolerances(torch.float64))
    torch.testing.assert_close(mps_mpo_expectation(mps, mpo), expected, **tolerances(torch.float64))
    # 全 cut で left[k] と right[k] を縮約しても同じ scalar になる（途中の environment の整合）。
    for cut in range(len(dims) + 1):
        torch.testing.assert_close(
            (left[cut] * right[cut]).sum(), expected, **tolerances(torch.float64)
        )
    # 境界
    assert left[0].shape == right[-1].shape == (1, 1, 1)
    assert left[0].item() == right[-1].item() == 1.0


@pytest.mark.parametrize("rank", [1, 2, 4])
def test_environment_works_for_rank_one_and_larger(rank):
    """(17) MPS の bond 次元 1 と 2 以上の両方で dense と一致する。"""
    mps, mpo, dense = _problem([2, 3, 2, 3], max_bond=rank)
    psi = dense_state(mps)
    torch.testing.assert_close(
        mps_mpo_expectation(mps, mpo), psi @ dense @ psi, **tolerances(torch.float64)
    )


def test_incremental_updates_match_full_builders():
    mps, mpo, _ = _problem([2, 3, 2, 3])
    left = build_left_environments(mps, mpo)
    environment = left[0]
    for site in range(len(mps)):
        environment = update_left_environment(environment, mps[site], mpo[site])
        torch.testing.assert_close(environment, left[site + 1])
    right = build_right_environments(mps, mpo)
    environment = right[-1]
    for site in reversed(range(len(mps))):
        environment = update_right_environment(environment, mps[site], mpo[site])
        torch.testing.assert_close(environment, right[site])


def test_expectation_and_energy_differ_for_unnormalized_state():
    """(3) 非正規化 MPS では期待値は ||psi||^2 倍され、energy（Rayleigh 商）は変わらない。"""
    mps, mpo, dense = _problem([2, 3, 2, 3])
    psi = dense_state(mps)
    norm_squared = psi @ psi
    assert abs(float(norm_squared) - 1.0) > 1e-3  # 非正規化であること
    expectation = mps_mpo_expectation(mps, mpo)
    energy = mps_mpo_energy(mps, mpo)
    torch.testing.assert_close(expectation, energy * norm_squared, **tolerances(torch.float64))
    torch.testing.assert_close(energy, psi @ dense @ psi / norm_squared, **tolerances(torch.float64))
    assert abs(float(expectation) - float(energy)) > 1e-6

    scaled = [core.clone() for core in mps]
    scaled[0] = scaled[0] * 3.0
    torch.testing.assert_close(mps_mpo_energy(scaled, mpo), energy, **tolerances(torch.float64))
    torch.testing.assert_close(mps_mpo_expectation(scaled, mpo), 9.0 * expectation, **tolerances(torch.float64))


def test_energy_works_with_tfim_mpo_and_is_not_tfim_specific():
    mpo = build_tfim_mpo(5, 1.0, 0.8, dtype=torch.float64, device="cpu")
    dense = build_tfim_dense_hamiltonian(5, 1.0, 0.8, dtype=torch.float64, device="cpu")
    mps = random_mps([2] * 5, 3)
    psi = dense_state(mps)
    torch.testing.assert_close(
        mps_mpo_energy(mps, mpo), psi @ dense @ psi / (psi @ psi), **tolerances(torch.float64)
    )


def test_normalize_mps_gives_unit_norm_without_mutation():
    """(4)(22) 正規化後のノルムが 1。scale は core 0 だけに吸収され、入力は変わらない。"""
    mps, _, _ = _problem([2, 3, 4, 2])
    original = copy.deepcopy(mps)
    normalized = normalize_mps(mps)
    for before, after in zip(original, mps):
        assert torch.equal(before, after)
    torch.testing.assert_close(tt_fro_norm(normalized), torch.tensor(1.0, dtype=torch.float64))
    psi = dense_state(normalized)
    torch.testing.assert_close(psi @ psi, torch.tensor(1.0, dtype=torch.float64))
    assert not torch.equal(normalized[0], mps[0])
    for site in range(1, len(mps)):
        assert torch.equal(normalized[site], mps[site])
        assert normalized[site].data_ptr() != mps[site].data_ptr()
    # 状態の向きは変わらない
    ratio = dense_state(normalized) / dense_state(mps)
    torch.testing.assert_close(ratio, ratio[0].expand_as(ratio))


def test_normalize_mps_rejects_zero_norm():
    zero = [torch.zeros(1, 2, 2, dtype=torch.float64), torch.zeros(2, 2, 1, dtype=torch.float64)]
    with pytest.raises(ValueError, match="正規化できません"):
        normalize_mps(zero)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", available_devices(), ids=str)
def test_dtype_device_preserved_and_matches_dense(dtype, device):
    """(19)(20)(21)"""
    mps, mpo, dense = _problem([2, 3, 2, 3], dtype=dtype, device=device)
    psi = dense_state(mps)
    for environments in (build_left_environments(mps, mpo), build_right_environments(mps, mpo)):
        assert all(env.dtype == dtype and env.device.type == device.type for env in environments)
    energy = mps_mpo_energy(mps, mpo)
    assert energy.dtype == dtype and energy.device.type == device.type and energy.shape == ()
    assert normalize_mps(mps)[0].dtype == dtype
    torch.testing.assert_close(energy, psi @ dense @ psi / (psi @ psi), **tolerances(dtype))


def test_inputs_are_not_mutated():
    """(22)"""
    mps, mpo, _ = _problem([2, 3, 2])
    mps_before, mpo_before = copy.deepcopy(mps), copy.deepcopy(mpo)
    build_left_environments(mps, mpo)
    build_right_environments(mps, mpo)
    mps_mpo_expectation(mps, mpo)
    mps_mpo_energy(mps, mpo)
    normalize_mps(mps)
    for a, b in zip(mps + mpo, mps_before + mpo_before):
        assert torch.equal(a, b)


def test_energy_backward_matches_dense_autograd():
    """(26) 低レベル API の backward が有限で、dense 経由の勾配と一致する。"""
    dims = [2, 3, 2]
    mpo, dense = random_chain_hamiltonian(dims)
    mps = [core.clone().requires_grad_(True) for core in random_mps(dims, 3, seed=5)]
    mps_mpo_energy(mps, mpo).backward()
    grads = [core.grad.clone() for core in mps]
    assert all(torch.isfinite(grad).all() for grad in grads)

    reference = [core.detach().clone().requires_grad_(True) for core in mps]
    psi = dense_state(reference)
    (psi @ dense @ psi / (psi @ psi)).backward()
    for grad, ref in zip(grads, reference):
        torch.testing.assert_close(grad, ref.grad, atol=1e-9, rtol=1e-8)


def test_environment_update_backward_is_finite():
    mps = [core.clone().requires_grad_(True) for core in random_mps([2, 3], 2)]
    mpo, _ = random_chain_hamiltonian([2, 3])
    environments = build_left_environments(mps, mpo)
    environments[-1].sum().backward()
    assert all(core.grad is not None and torch.isfinite(core.grad).all() for core in mps)


def _valid_pair():
    mps, mpo, _ = _problem([2, 3, 2])
    return mps, mpo


def test_rejects_site_count_mismatch_before_truncating_with_zip():
    mps, mpo = _valid_pair()
    with pytest.raises(ValueError, match="site 数"):
        mps_mpo_energy(mps[:-1], mpo)
    with pytest.raises(ValueError, match="site 数"):
        build_left_environments(mps, mpo[:-1])


@pytest.mark.parametrize("builder", [build_left_environments, build_right_environments, mps_mpo_energy])
def test_rejects_invalid_structures(builder):
    """(23) shape / bond / 物理次元 / dtype / device / 空 / 有限性。"""
    mps, mpo = _valid_pair()
    with pytest.raises(ValueError):
        builder([], mpo)
    with pytest.raises(ValueError):
        builder(mps, [])
    with pytest.raises(ValueError, match="3階"):
        builder([mps[0].reshape(2, 2), *mps[1:]], mpo)
    with pytest.raises(ValueError, match="4階"):
        builder(mps, [mpo[0].reshape(-1, 2, 2), *mpo[1:]])
    with pytest.raises(ValueError, match="left bond"):
        builder([mps[0].expand(2, -1, -1).clone(), *mps[1:]], mpo)
    # 隣接 bond の不一致
    broken = [mps[0], torch.zeros(5, 3, 2, dtype=torch.float64), mps[2]]
    with pytest.raises(ValueError, match="bond"):
        builder(broken, mpo)
    # MPO の物理次元が MPS と違う
    with pytest.raises(ValueError, match="物理次元"):
        builder(mps, [mpo[0], mpo[1][:, :2, :2], mpo[2]])
    bad_in = mpo[1][:, :, :2, :]
    with pytest.raises(ValueError, match="物理次元"):
        builder(mps, [mpo[0], bad_in, mpo[2]])
    # dtype
    with pytest.raises(TypeError):
        builder([core.float() for core in mps], mpo)
    # device（meta）
    with pytest.raises(ValueError, match="device"):
        builder([*mps[:-1], mps[-1].to("meta")], mpo)
    # 有限性
    nan_mps = [core.clone() for core in mps]
    nan_mps[1][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN / inf"):
        builder(nan_mps, mpo)
    inf_mpo = [core.clone() for core in mpo]
    inf_mpo[0][0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="NaN / inf"):
        builder(mps, inf_mpo)


@pytest.mark.parametrize(
    "dtype",
    [torch.complex128, torch.float16, torch.bfloat16, torch.int64, torch.bool],
)
def test_rejects_unsupported_dtypes(dtype):
    mps, mpo = _valid_pair()
    bad_mps = [torch.ones(core.shape, dtype=dtype) for core in mps]
    bad_mpo = [torch.ones(core.shape, dtype=dtype) for core in mpo]
    with pytest.raises(TypeError):
        mps_mpo_energy(bad_mps, bad_mpo)
    with pytest.raises(TypeError):
        update_left_environment(torch.ones(1, 1, 1, dtype=dtype), bad_mps[0], bad_mpo[0])
    with pytest.raises(TypeError):
        normalize_mps(bad_mps)


def test_single_core_update_validation():
    mps, mpo = _valid_pair()
    left = build_left_environments(mps, mpo)
    right = build_right_environments(mps, mpo)
    with pytest.raises(ValueError, match="shape"):
        update_left_environment(left[1], mps[0], mpo[0])  # 取り込む site を取り違える
    with pytest.raises(ValueError, match="shape"):
        update_right_environment(right[1], mps[2], mpo[2])
    with pytest.raises(TypeError, match="dtype"):
        update_left_environment(left[0].float(), mps[0], mpo[0])
    with pytest.raises(ValueError, match="device"):
        update_left_environment(left[0].to("meta"), mps[0], mpo[0])
    with pytest.raises(ValueError, match="3階"):
        update_left_environment(left[0][0], mps[0], mpo[0])
    with pytest.raises(ValueError, match="物理次元"):
        update_left_environment(left[0], mps[0], mpo[0][:, :1, :1])


def test_zero_norm_mps_energy_is_rejected():
    mps, mpo = _valid_pair()
    zero = [torch.zeros_like(core) for core in mps]
    with pytest.raises(ValueError, match="ノルム"):
        mps_mpo_energy(zero, mpo)
    # 期待値は定義できる（0）
    assert mps_mpo_expectation(zero, mpo).item() == 0.0


def test_boundary_environment_follows_input_dtype_and_device():
    mps, mpo, _ = _problem([2, 2], dtype=torch.float32)
    assert build_left_environments(mps, mpo)[0].dtype == torch.float32
    assert build_right_environments(mps, mpo)[-1].dtype == torch.float32


def _inverse_gauge(mps, scale: float, bond: int) -> list[torch.Tensor]:
    """隣接 core を ``scale`` と ``1/scale`` で再スケールする。物理状態は不変。"""
    gauged = [core.clone() for core in mps]
    gauged[bond] = gauged[bond] * scale
    gauged[bond + 1] = gauged[bond + 1] / scale
    return gauged


@pytest.mark.parametrize("dtype,scale", [(torch.float32, 1e20), (torch.float64, 1e200)])
def test_energy_and_normalize_survive_inverse_bond_scale(dtype, scale):
    """隣接 bond 上の逆スケールでも energy・正規化・DMRG の初期 energy が有限。"""
    dims = [2, 3, 2, 3]
    mps = random_mps(dims, 2, seed=1, dtype=dtype)
    mpo, _ = random_chain_hamiltonian(dims, seed=2, dtype=dtype)
    gauged = _inverse_gauge(mps, scale, bond=1)
    close = {"atol": 1e-4, "rtol": 1e-4} if dtype == torch.float32 else {"atol": 1e-8, "rtol": 1e-8}
    torch.testing.assert_close(tt_reconstruct(gauged), tt_reconstruct(mps), **close)

    before = [core.clone() for core in gauged]
    mpo_before = [core.clone() for core in mpo]
    energy = mps_mpo_energy(mps, mpo)
    gauged_energy = mps_mpo_energy(gauged, mpo)
    assert torch.isfinite(gauged_energy) and gauged_energy.dtype == dtype
    torch.testing.assert_close(gauged_energy, energy, **close)

    normalized = normalize_mps(gauged)
    assert all(torch.isfinite(core).all() and core.dtype == dtype for core in normalized)
    torch.testing.assert_close(tt_fro_norm(normalized), torch.tensor(1.0, dtype=dtype), **close)
    reference_direction = tt_reconstruct(normalize_mps(mps))
    torch.testing.assert_close(tt_reconstruct(normalized), reference_direction, **close)
    for actual, original in zip(gauged, before):
        assert torch.equal(actual, original)
    for actual, original in zip(mpo, mpo_before):
        assert torch.equal(actual, original)

    energy_leaves = [core.detach().clone().requires_grad_(True) for core in gauged]
    mps_mpo_energy(energy_leaves, mpo).backward()
    assert all(leaf.grad is not None and torch.isfinite(leaf.grad).all() for leaf in energy_leaves)
    norm_leaves = [core.detach().clone().requires_grad_(True) for core in gauged]
    tt_reconstruct(normalize_mps(norm_leaves)).sum().backward()
    assert all(leaf.grad is not None and torch.isfinite(leaf.grad).all() for leaf in norm_leaves)

    result = run_two_site_dmrg(
        gauged,
        mpo,
        max_bond=None,
        min_round_trips=1,
        max_round_trips=1,
        energy_tolerance=1.0,
        required_consecutive=1,
        **sweep_tolerances(dtype),
    )
    assert result.history[0].energy == pytest.approx(float(energy), rel=1e-4, abs=1e-4)
