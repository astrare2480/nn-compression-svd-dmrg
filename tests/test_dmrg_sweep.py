"""``dmrg.sweep``（2-site sweep 1 方向）のテスト。"""

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
from nn_compression.compression import tt_canonicality_errors, tt_ranks
from nn_compression.dmrg import (
    SweepPassResult,
    SweepStepLog,
    build_tfim_dense_hamiltonian,
    build_tfim_mpo,
    mps_mpo_energy,
    two_site_sweep,
)

DIRECTIONS = ("left_to_right", "right_to_left")
CASES = {
    "two_sites": ([3, 2], 2),
    "d2": ([2, 2, 2, 2, 2], 2),
    "d3": ([3, 3, 3, 3], 3),
    "d5": ([5, 4, 5], 4),
    "mixed_rank1": ([2, 3, 4, 2, 3], 1),
    "mixed_rank3": ([2, 3, 4, 2, 3], 3),
}
TOL = sweep_tolerances(torch.float64)


def _problem(dims, max_bond, *, dtype=torch.float64, device="cpu"):
    mpo, dense = random_chain_hamiltonian(dims, seed=21, dtype=dtype, device=device)
    mps = random_mps(dims, max_bond, seed=22, dtype=dtype, device=device)
    return mps, mpo, dense


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("dims,max_bond", CASES.values(), ids=CASES.keys())
def test_sweep_logs_canonical_form_and_energy(dims, max_bond, direction):
    """(10)(11)(16)(17)(18) ログ・正準形・energy（切り捨てなし）。"""
    mps, mpo, dense = _problem(dims, max_bond)
    num_sites = len(dims)
    result = two_site_sweep(mps, mpo, direction=direction, max_bond=None, **TOL)
    assert isinstance(result, SweepPassResult) and len(result.logs) == num_sites - 1
    assert all(isinstance(log, SweepStepLog) for log in result.logs)

    expected_order = list(range(num_sites - 1))
    if direction == "right_to_left":
        expected_order.reverse()
    assert [log.pair_start for log in result.logs] == expected_order
    assert all(log.direction == direction for log in result.logs)

    # 正準形: ltr は終了時に center=d-1、rtl は center=0。
    if direction == "left_to_right":
        errors = tt_canonicality_errors(result.cores, center=num_sites - 1)["left"]
    else:
        errors = tt_canonicality_errors(result.cores, center=0)["right"]
    assert all(error.item() <= 1e-10 for error in errors)

    # 切り捨てなし: 各更新後の全体 energy は局所最小値に一致し、単調に減少する。
    energies = [log.energy_after.item() for log in result.logs]
    initial = mps_mpo_energy(mps, mpo).item()
    sequence = [initial, *energies]
    assert all(later <= earlier + 1e-10 for earlier, later in zip(sequence, sequence[1:]))
    for log in result.logs:
        torch.testing.assert_close(log.energy_after, log.local_minimum_energy, **tolerances(torch.float64))
        assert log.local_residual.item() <= 1e-10
        assert log.discarded_weight.item() == 0.0
    # 最終 log の energy は返された MPS の energy であり、dense 計算と一致する。
    psi = dense_state(result.cores)
    torch.testing.assert_close(psi @ psi, torch.tensor(1.0, dtype=torch.float64), **tolerances(torch.float64))
    torch.testing.assert_close(result.logs[-1].energy_after, psi @ dense @ psi, **tolerances(torch.float64))
    # 変分原理
    assert energies[-1] >= torch.linalg.eigvalsh(dense)[0].item() - 1e-10


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_truncated_sweep_respects_bond_cap_and_records_logs(direction):
    dims = [3, 3, 3, 3, 3]
    mps, mpo, dense = _problem(dims, 4)
    result = two_site_sweep(mps, mpo, direction=direction, max_bond=2, **TOL)
    assert max(tt_ranks(result.cores)[1:-1]) <= 2
    assert all(log.retained_rank <= 2 for log in result.logs)
    assert any(log.discarded_weight.item() > 0 for log in result.logs)
    # 切り捨て後 energy は dense と一致し、変分下限以上。
    psi = dense_state(result.cores)
    final = result.logs[-1].energy_after
    torch.testing.assert_close(final, psi @ dense @ psi / (psi @ psi), **tolerances(torch.float64))
    assert final.item() >= torch.linalg.eigvalsh(dense)[0].item() - 1e-10
    # 切り捨てでは local_minimum と energy_after は別の量になり得る
    assert all(log.energy_after.item() >= log.local_minimum_energy.item() - 1e-10 for log in result.logs)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_sweep_accepts_unnormalized_non_canonical_input_and_tfim(direction):
    """入力は正規化・正準化されていなくてよい。TFIM でも非 TFIM でも動く。"""
    mpo = build_tfim_mpo(6, 1.0, 0.9, dtype=torch.float64, device="cpu")
    dense = build_tfim_dense_hamiltonian(6, 1.0, 0.9, dtype=torch.float64, device="cpu")
    mps = random_mps([2] * 6, 3, seed=5)
    mps[2] = mps[2] * 7.0
    result = two_site_sweep(mps, mpo, direction=direction, max_bond=None, **TOL)
    psi = dense_state(result.cores)
    torch.testing.assert_close(psi @ psi, torch.tensor(1.0, dtype=torch.float64), **tolerances(torch.float64))
    assert result.logs[-1].energy_after.item() <= mps_mpo_energy(mps, mpo).item() + 1e-10
    torch.testing.assert_close(result.logs[-1].energy_after, psi @ dense @ psi, **tolerances(torch.float64))


def test_sweep_does_not_mutate_inputs_and_returns_independent_cores():
    """(22)"""
    mps, mpo, _ = _problem([2, 3, 2, 3], 2)
    before = copy.deepcopy([*mps, *mpo])
    for direction in DIRECTIONS:
        result = two_site_sweep(mps, mpo, direction=direction, max_bond=None, **TOL)
        for core in result.cores:
            assert all(core.data_ptr() != other.data_ptr() for other in (*mps, *mpo))
    for a, b in zip([*mps, *mpo], before):
        assert torch.equal(a, b)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", available_devices(), ids=str)
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_sweep_dtype_device_preserved(dtype, device, direction):
    """(19)(20)(21)"""
    mps, mpo, dense = _problem([2, 3, 2, 3], 2, dtype=dtype, device=device)
    result = two_site_sweep(mps, mpo, direction=direction, max_bond=None, **sweep_tolerances(dtype))
    for core in result.cores:
        assert core.dtype == dtype and core.device.type == device.type
    for log in result.logs:
        for tensor in (log.local_minimum_energy, log.local_residual, log.discarded_weight, log.energy_after):
            assert tensor.dtype == dtype and tensor.device.type == device.type
    psi = dense_state(result.cores)
    torch.testing.assert_close(
        result.logs[-1].energy_after, psi @ dense @ psi / (psi @ psi), **tolerances(dtype)
    )


def test_sweep_accepts_tuples():
    mps, mpo, _ = _problem([2, 3, 2], 2)
    result = two_site_sweep(tuple(mps), tuple(mpo), direction="left_to_right", max_bond=None, **TOL)
    assert len(result.cores) == 3


def test_sweep_rejects_invalid_arguments():
    """(23)(24)"""
    mps, mpo, _ = _problem([2, 3, 2, 3], 2)
    good = {"direction": "left_to_right", "max_bond": None, **TOL}
    for direction in ("left", "right", "rtl", "", None, 1):
        with pytest.raises((TypeError, ValueError)):
            two_site_sweep(mps, mpo, **{**good, "direction": direction})
    for bad in (True, False, 0, -2, 1.5, "3"):
        with pytest.raises((TypeError, ValueError)):
            two_site_sweep(mps, mpo, **{**good, "max_bond": bad})
    for name in ("atol", "rtol"):
        for bad in (-1.0, float("nan"), float("inf"), True, "1e-8", None):
            with pytest.raises((TypeError, ValueError)):
                two_site_sweep(mps, mpo, **{**good, name: bad})
    with pytest.raises(ValueError, match="site 数"):
        two_site_sweep(mps[:-1], mpo, **good)
    with pytest.raises(ValueError, match="2 site"):
        two_site_sweep(
            [torch.ones(1, 2, 1, dtype=torch.float64)],
            [torch.ones(1, 2, 2, 1, dtype=torch.float64)],
            **good,
        )
    with pytest.raises(TypeError):
        two_site_sweep([c.float() for c in mps], mpo, **good)
    with pytest.raises(ValueError, match="物理次元"):
        two_site_sweep(mps, [mpo[0], mpo[1][:, :2, :2], mpo[2], mpo[3]], **good)
    bad_mps = [core.clone() for core in mps]
    bad_mps[1][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN / inf"):
        two_site_sweep(bad_mps, mpo, **good)


def test_unreachable_local_tolerance_raises_runtime_error():
    """局所残差が許容を超えたら黙って続行せず ``RuntimeError``。"""
    mps, mpo, _ = _problem([2, 3, 2, 3], 2, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="局所残差"):
        two_site_sweep(mps, mpo, direction="left_to_right", max_bond=None, atol=1e-12, rtol=0.0)


def test_non_hermitian_mpo_is_rejected():
    mps, mpo, _ = _problem([2, 3, 2, 3], 2)
    skewed = [core.clone() for core in mpo]
    skewed[1] = skewed[1] + 0.5 * torch.randn_like(skewed[1])
    with pytest.raises(ValueError, match="対称"):
        two_site_sweep(mps, skewed, direction="left_to_right", max_bond=None, **TOL)
