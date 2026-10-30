"""``dmrg.convergence``（往復と停止判定）のテスト。"""

from __future__ import annotations

import copy
import math
from unittest import mock

import pytest
import torch

from _dmrg_testing import (
    DTYPES,
    available_devices,
    random_chain_hamiltonian,
    random_mps,
    sweep_tolerances,
    tolerances,
)
from nn_compression.compression import tt_canonicality_errors, tt_ranks
from nn_compression.dmrg import (
    DMRGResult,
    DMRGRoundTripRecord,
    DMRGRoundTripResult,
    mps_mpo_energy,
    run_two_site_dmrg,
    run_two_site_dmrg_round_trip,
)
from nn_compression.dmrg.convergence import _choose_stop_reason, _next_stability_count

TOL = sweep_tolerances(torch.float64)
CASES = {
    "two_sites": ([3, 2], 2),
    "d2": ([2, 2, 2, 2, 2], 2),
    "d3": ([3, 3, 3, 3], 3),
    "mixed": ([2, 3, 4, 2, 3], 2),
}


def _problem(dims, max_bond, *, dtype=torch.float64, device="cpu"):
    mpo, dense = random_chain_hamiltonian(dims, seed=31, dtype=dtype, device=device)
    mps = random_mps(dims, max_bond, seed=32, dtype=dtype, device=device)
    return mps, mpo, dense


def _settings(**overrides):
    settings = {
        "max_bond": None,
        "min_round_trips": 2,
        "max_round_trips": 12,
        "energy_tolerance": 1e-10,
        "required_consecutive": 2,
        **TOL,
    }
    settings.update(overrides)
    return settings


@pytest.mark.parametrize("dims,max_bond", CASES.values(), ids=CASES.keys())
def test_round_trip_log_count_order_and_final_canonical_form(dims, max_bond):
    """(10)(12)(16)(18)"""
    mps, mpo, _ = _problem(dims, max_bond)
    num_sites = len(dims)
    result = run_two_site_dmrg_round_trip(mps, mpo, max_bond=None, **TOL)
    assert isinstance(result, DMRGRoundTripResult)
    assert len(result.logs) == 2 * (num_sites - 1)
    forward, backward = result.logs[: num_sites - 1], result.logs[num_sites - 1 :]
    assert [log.pair_start for log in forward] == list(range(num_sites - 1))
    assert [log.pair_start for log in backward] == list(reversed(range(num_sites - 1)))
    assert all(log.direction == "left_to_right" for log in forward)
    assert all(log.direction == "right_to_left" for log in backward)
    errors = tt_canonicality_errors(result.cores, center=0)["right"]
    assert all(error.item() <= 1e-10 for error in errors)
    assert result.logs[-1].energy_after.item() <= mps_mpo_energy(mps, mpo).item() + 1e-10


def test_round_trip_does_not_mutate_inputs():
    """(22)"""
    mps, mpo, _ = _problem([2, 3, 2, 3], 2)
    before = copy.deepcopy([*mps, *mpo])
    run_two_site_dmrg_round_trip(mps, mpo, max_bond=2, **TOL)
    for a, b in zip([*mps, *mpo], before):
        assert torch.equal(a, b)


# ---- 停止判定の部品 -----------------------------------------------------------------


def test_next_stability_count_increments_resets_and_handles_boundary():
    """(13) 連続回数の増加・リセット・境界（<=）・未定義値。"""
    tolerance = 1e-10
    assert _next_stability_count(0.5 * tolerance, tolerance=tolerance, current_count=0) == 1
    assert _next_stability_count(0.5 * tolerance, tolerance=tolerance, current_count=1) == 2
    assert _next_stability_count(2.0 * tolerance, tolerance=tolerance, current_count=2) == 0
    assert _next_stability_count(tolerance, tolerance=tolerance, current_count=0) == 1  # 境界は安定
    assert _next_stability_count(math.nextafter(tolerance, 1.0), tolerance=tolerance, current_count=3) == 0
    assert _next_stability_count(None, tolerance=tolerance, current_count=2) == 0
    for bad in (float("nan"), float("inf")):
        with pytest.raises(FloatingPointError):
            _next_stability_count(bad, tolerance=tolerance, current_count=0)
    with pytest.raises(ValueError):
        _next_stability_count(0.0, tolerance=-1.0, current_count=0)
    with pytest.raises(ValueError):
        _next_stability_count(0.0, tolerance=1.0, current_count=-1)


def test_choose_stop_reason_branches():
    """(13) 最小往復数・連続回数・最大往復数・同時成立。"""
    kwargs = {"min_round_trips": 2, "max_round_trips": 12, "required_consecutive": 2}
    assert _choose_stop_reason(round_trip=1, stable_count=2, **kwargs) == "continue"
    assert _choose_stop_reason(round_trip=2, stable_count=1, **kwargs) == "continue"
    assert _choose_stop_reason(round_trip=2, stable_count=2, **kwargs) == "converged"
    assert _choose_stop_reason(round_trip=12, stable_count=2, **kwargs) == "converged"  # 同時成立は converged
    assert _choose_stop_reason(round_trip=12, stable_count=0, **kwargs) == "max_round_trips_reached"
    assert _choose_stop_reason(round_trip=11, stable_count=0, **kwargs) == "continue"
    with pytest.raises(ValueError):
        _choose_stop_reason(round_trip=1, stable_count=0, min_round_trips=3, max_round_trips=2, required_consecutive=1)
    with pytest.raises(ValueError):
        _choose_stop_reason(round_trip=1, stable_count=0, min_round_trips=0, max_round_trips=2, required_consecutive=1)
    with pytest.raises(ValueError):
        _choose_stop_reason(round_trip=1, stable_count=0, min_round_trips=1, max_round_trips=2, required_consecutive=0)


# ---- 高レベル DMRG ------------------------------------------------------------------


@pytest.mark.parametrize("dims,max_bond", CASES.values(), ids=CASES.keys())
def test_dmrg_converges_to_exact_ground_state_without_dense_inputs(dims, max_bond):
    """(14)(15)(16)(18) dense Hamiltonian・厳密 energy なしで動き、非 TFIM の小規模系で基底状態へ収束する。"""
    mps, mpo, dense = _problem(dims, max_bond)
    result = run_two_site_dmrg(mps, mpo, **_settings())
    assert isinstance(result, DMRGResult) and result.stop_reason == "converged"
    exact = torch.linalg.eigvalsh(dense)[0].item()
    assert result.history[-1].energy == pytest.approx(exact, abs=1e-8)
    torch.testing.assert_close(
        mps_mpo_energy(result.cores, mpo), torch.tensor(result.history[-1].energy, dtype=torch.float64)
    )
    assert all(not core.requires_grad for core in result.cores)


def test_history_structure_and_converged_stop():
    """(13) 初期状態 round_trip=0 を含む履歴と、converged の条件。"""
    mps, mpo, _ = _problem([2, 3, 4, 2, 3], 2)
    result = run_two_site_dmrg(mps, mpo, **_settings())
    history = result.history
    assert [record.round_trip for record in history] == list(range(len(history)))
    first = history[0]
    assert isinstance(first, DMRGRoundTripRecord)
    assert first.energy_change is None and first.max_discarded_weight is None and first.stability_count == 0
    assert first.energy == pytest.approx(mps_mpo_energy(mps, mpo).item())
    assert first.actual_bond_dimensions == tuple(tt_ranks(mps)[1:-1])
    for record in history[1:]:
        assert isinstance(record.energy, float) and math.isfinite(record.energy)
        assert isinstance(record.energy_change, float) and record.energy_change >= 0.0
        assert isinstance(record.max_discarded_weight, float)
        assert isinstance(record.stability_count, int)
        assert len(record.actual_bond_dimensions) == len(mps) - 1
    for previous, record in zip(history, history[1:]):
        assert record.energy_change == pytest.approx(abs(record.energy - previous.energy), abs=1e-15)
        expected_count = previous.stability_count + 1 if record.energy_change <= 1e-10 else 0
        assert record.stability_count == expected_count
    # converged なら、最後の required_consecutive 回はすべて energy_change <= tolerance
    required = 2
    tail = history[-required:]
    assert all(record.energy_change <= 1e-10 for record in tail)
    assert history[-1].round_trip >= 2
    assert history[-1].stability_count >= required
    # 最後より前では停止条件を満たしていない
    for record in history[:-1]:
        assert not (record.round_trip >= 2 and record.stability_count >= required)


def test_max_round_trips_reached_when_not_stable():
    mps, mpo, _ = _problem([3, 3, 3, 3], 3)
    result = run_two_site_dmrg(
        mps, mpo, **_settings(max_bond=2, min_round_trips=1, max_round_trips=2, energy_tolerance=1e-30)
    )
    assert result.stop_reason == "max_round_trips_reached"
    assert len(result.history) == 3 and result.history[-1].round_trip == 2
    assert result.history[-1].stability_count < 2


def test_converged_takes_priority_when_stability_and_limit_coincide():
    mps, mpo, _ = _problem([3, 3, 3, 3], 3)
    result = run_two_site_dmrg(
        mps, mpo, **_settings(min_round_trips=2, max_round_trips=2, energy_tolerance=1e10)
    )
    assert result.stop_reason == "converged" and result.history[-1].round_trip == 2


def test_min_round_trips_delays_stop_and_required_consecutive_is_respected():
    mps, mpo, _ = _problem([3, 3, 3, 3], 3)
    huge = {"energy_tolerance": 1e10}
    result = run_two_site_dmrg(mps, mpo, **_settings(min_round_trips=4, max_round_trips=9, required_consecutive=1, **huge))
    assert result.history[-1].round_trip == 4 and result.stop_reason == "converged"
    result = run_two_site_dmrg(mps, mpo, **_settings(min_round_trips=1, max_round_trips=9, required_consecutive=3, **huge))
    assert result.history[-1].round_trip == 3 and result.history[-1].stability_count == 3


def test_energy_tolerance_boundary_is_inclusive():
    """energy_change == energy_tolerance は安定条件を満たす（<=）。"""
    mps, mpo, _ = _problem([3, 3, 3, 3], 3)
    probe = run_two_site_dmrg(mps, mpo, **_settings(min_round_trips=1, max_round_trips=1, required_consecutive=1))
    change = probe.history[1].energy_change
    assert change > 0.0
    at_boundary = run_two_site_dmrg(
        mps, mpo, **_settings(min_round_trips=1, max_round_trips=3, required_consecutive=1, energy_tolerance=change)
    )
    assert at_boundary.history[-1].round_trip == 1 and at_boundary.stop_reason == "converged"
    below = run_two_site_dmrg(
        mps,
        mpo,
        **_settings(
            min_round_trips=1, max_round_trips=1, required_consecutive=1, energy_tolerance=math.nextafter(change, 0.0)
        ),
    )
    assert below.stop_reason == "max_round_trips_reached"


@pytest.mark.parametrize("max_bond", [1, 2, 3])
def test_bond_cap_respected_in_every_round(max_bond):
    mps, mpo, dense = _problem([3, 3, 3, 3, 3], 3)
    result = run_two_site_dmrg(mps, mpo, **_settings(max_bond=max_bond, max_round_trips=6))
    for record in result.history[1:]:
        assert max(record.actual_bond_dimensions) <= max_bond
    assert result.history[-1].energy >= torch.linalg.eigvalsh(dense)[0].item() - 1e-10
    assert all(record.max_discarded_weight >= 0.0 for record in result.history[1:])


def test_dmrg_does_not_mutate_inputs():
    """(22)"""
    mps, mpo, _ = _problem([2, 3, 2, 3], 2)
    before = copy.deepcopy([*mps, *mpo])
    run_two_site_dmrg(mps, mpo, **_settings(max_round_trips=3))
    for a, b in zip([*mps, *mpo], before):
        assert torch.equal(a, b)


def test_dmrg_runs_without_building_dense_or_model_specific_objects():
    """(14)(29) DMRG 実行中に dense 化・dense 診断・TFIM builder が呼ばれない。"""
    mps, mpo, _ = _problem([2, 3, 2, 3], 2)

    def forbidden(*args, **kwargs):
        raise AssertionError("DMRG 本体から dense / 模型固有の処理が呼ばれました。")

    targets = [
        "nn_compression.dmrg.diagnostics.mpo_to_dense",
        "nn_compression.dmrg.diagnostics.compute_dense_residual_and_variance",
        "nn_compression.dmrg.models.tfim.build_tfim_mpo",
        "nn_compression.dmrg.models.tfim.build_tfim_dense_hamiltonian",
        "nn_compression.dmrg.initialization.make_random_mps",
        "nn_compression.compression.tt_reconstruct",
        "nn_compression.compression.tt.tt_reconstruct",
    ]
    patches = [mock.patch(target, side_effect=forbidden) for target in targets]
    for patch in patches:
        patch.start()
    try:
        result = run_two_site_dmrg(mps, mpo, **_settings(max_round_trips=4))
    finally:
        for patch in patches:
            patch.stop()
    assert result.history[-1].round_trip >= 1


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", available_devices(), ids=str)
def test_dmrg_dtype_device_preserved(dtype, device):
    """(19)(20)(21)"""
    mps, mpo, dense = _problem([2, 3, 2, 3], 2, dtype=dtype, device=device)
    result = run_two_site_dmrg(
        mps, mpo, **_settings(max_round_trips=8, energy_tolerance=1e-5 if dtype == torch.float32 else 1e-10, **sweep_tolerances(dtype))
    )
    for core in result.cores:
        assert core.dtype == dtype and core.device.type == device.type
    exact = torch.linalg.eigvalsh(dense.double())[0].item()
    assert result.history[-1].energy == pytest.approx(exact, abs=tolerances(dtype)["atol"] * 10)


def test_dmrg_ignores_requires_grad_of_inputs():
    mps, mpo, _ = _problem([2, 3, 2], 2)
    mps = [core.clone().requires_grad_(True) for core in mps]
    result = run_two_site_dmrg(mps, mpo, **_settings(max_round_trips=3))
    assert all(not core.requires_grad for core in result.cores)


def test_invalid_settings_are_rejected():
    """(23)(24)"""
    mps, mpo, _ = _problem([2, 3, 2, 3], 2)

    def run(**overrides):
        return run_two_site_dmrg(mps, mpo, **_settings(**overrides))

    for name in ("min_round_trips", "max_round_trips", "required_consecutive"):
        for bad in (True, False, 0, -1, 1.5, "2"):
            with pytest.raises((TypeError, ValueError)):
                run(**{name: bad})
    with pytest.raises(ValueError, match="min_round_trips"):
        run(min_round_trips=5, max_round_trips=4)
    for bad in (-1e-3, float("nan"), float("inf"), True, "1e-8", None):
        with pytest.raises((TypeError, ValueError)):
            run(energy_tolerance=bad)
        with pytest.raises((TypeError, ValueError)):
            run(atol=bad)
        with pytest.raises((TypeError, ValueError)):
            run(rtol=bad)
    for bad in (True, 0, -3, 2.0):
        with pytest.raises((TypeError, ValueError)):
            run(max_bond=bad)
    with pytest.raises(ValueError, match="site 数"):
        run_two_site_dmrg(mps[:-1], mpo, **_settings())
    with pytest.raises(ValueError, match="2 site"):
        run_two_site_dmrg(
            [torch.ones(1, 2, 1, dtype=torch.float64)],
            [torch.ones(1, 2, 2, 1, dtype=torch.float64)],
            **_settings(),
        )
    with pytest.raises(ValueError, match="ノルム"):
        run_two_site_dmrg([torch.zeros_like(core) for core in mps], mpo, **_settings())
    with pytest.raises(TypeError):
        run_two_site_dmrg([c.to(torch.complex128) for c in mps], mpo, **_settings())
