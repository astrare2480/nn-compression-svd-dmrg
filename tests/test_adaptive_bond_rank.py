"""adaptive bond rank の選択、固定 rank 経路、正準条件、可変 shape の復元。"""

from __future__ import annotations

import copy
import math

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from nn_compression.compression import TTLinear, assign_tt_linear_cores
from nn_compression.models import MNISTMLP
from nn_compression.training import (
    GpuResidentTensorBatches,
    update_tt_linear_two_core_pair,
)
from nn_compression.tt_mps import (
    canonicalize_tt_matrix_cores,
    dense_to_tt_matrix_cores,
    merge_tt_matrix_cores,
    replace_tt_matrix_core_pair,
    select_bond_rank_from_singular_values,
    split_merged_tt_matrix,
    split_merged_tt_matrix_by_discarded_weight,
    tt_matrix_canonicality_errors,
    tt_matrix_ranks,
    tt_matrix_to_dense,
)

OUT_MODES = (4, 2, 2, 2, 4, 4)
IN_MODES = (2, 2, 2, 2, 7, 7)


def _spectrum() -> torch.Tensor:
    return torch.tensor([math.sqrt(0.5), math.sqrt(0.3), math.sqrt(0.2)], dtype=torch.float64)


def test_threshold_boundary_selects_smallest_rank() -> None:
    decision = select_bond_rank_from_singular_values(
        _spectrum(),
        threshold=0.2,
        min_bond_rank=1,
        max_bond_rank=12,
    )
    assert decision.available_rank == 3
    assert decision.threshold_required_rank == 2
    assert decision.retained_rank == 2
    assert decision.threshold_satisfied is True
    assert decision.relative_discarded_weight == pytest.approx(0.2)
    assert decision.cap_limited is False
    assert decision.min_rank_limited is False
    assert decision.zero_spectrum is False


def test_cap_keeps_max_rank_and_does_not_claim_the_threshold() -> None:
    decision = select_bond_rank_from_singular_values(
        _spectrum(),
        threshold=1e-6,
        min_bond_rank=1,
        max_bond_rank=1,
    )
    assert decision.threshold_required_rank == 3
    assert decision.retained_rank == 1
    assert decision.cap_limited is True
    assert decision.threshold_satisfied is False
    assert decision.relative_discarded_weight == pytest.approx(0.5)


def test_min_rank_can_exceed_the_threshold_rank() -> None:
    decision = select_bond_rank_from_singular_values(
        _spectrum(),
        threshold=1.0,
        min_bond_rank=2,
        max_bond_rank=12,
    )
    assert decision.threshold_required_rank == 1
    assert decision.retained_rank == 2
    assert decision.min_rank_limited is True
    assert decision.threshold_satisfied is True


def test_zero_spectrum_is_rank_one_with_zero_discard() -> None:
    decision = select_bond_rank_from_singular_values(
        torch.zeros(4),
        threshold=1e-4,
        min_bond_rank=1,
        max_bond_rank=12,
    )
    assert decision.retained_rank == 1
    assert decision.threshold_required_rank == 1
    assert decision.relative_discarded_weight == 0.0
    assert decision.zero_spectrum is True
    assert decision.threshold_satisfied is True
    assert decision.cap_limited is False
    assert decision.min_rank_limited is False


def test_zero_spectrum_keeps_min_bond_rank_above_one() -> None:
    decision = select_bond_rank_from_singular_values(
        torch.zeros(4),
        threshold=1e-4,
        min_bond_rank=2,
        max_bond_rank=12,
    )
    assert decision.threshold_required_rank == 1
    assert decision.retained_rank == 2
    assert decision.min_rank_limited is True
    assert decision.relative_discarded_weight == 0.0
    assert decision.threshold_satisfied is True
    assert decision.zero_spectrum is True
    assert decision.cap_limited is False


def _direct_relative_discarded_weight(values: torch.Tensor, rank: int) -> float:
    squares = values.detach().to(dtype=torch.float64).square()
    return float(squares[rank:].sum() / squares.sum())


def test_ordinary_scale_keeps_expected_rank_and_direct_tail() -> None:
    values = _spectrum()
    expected = {1: 1, 2: 2, 3: 3}
    thresholds = {1: 1.0, 2: 0.2, 3: 0.0}
    for rank, threshold in thresholds.items():
        decision = select_bond_rank_from_singular_values(
            values,
            threshold=threshold,
            min_bond_rank=1,
            max_bond_rank=12,
        )
        assert decision.retained_rank == expected[rank]
        assert decision.relative_discarded_weight == pytest.approx(
            _direct_relative_discarded_weight(values, rank)
        )
        assert math.isfinite(decision.relative_discarded_weight)


def test_huge_finite_singular_values_stay_finite() -> None:
    decision = select_bond_rank_from_singular_values(
        torch.tensor([1e200, 1e200, 1e200], dtype=torch.float64),
        threshold=1.0 / 3.0,
        min_bond_rank=1,
        max_bond_rank=12,
    )
    assert decision.zero_spectrum is False
    assert decision.retained_rank == 2
    assert math.isfinite(decision.relative_discarded_weight)
    assert decision.relative_discarded_weight == pytest.approx(1.0 / 3.0)


def test_tiny_nonzero_singular_values_are_not_treated_as_zero() -> None:
    decision = select_bond_rank_from_singular_values(
        torch.tensor([1e-200, 1e-200, 1e-200], dtype=torch.float64),
        threshold=0.0,
        min_bond_rank=1,
        max_bond_rank=12,
    )
    assert decision.zero_spectrum is False
    assert decision.retained_rank == 3
    assert decision.relative_discarded_weight == 0.0
    assert math.isfinite(decision.relative_discarded_weight)


def test_threshold_zero_keeps_underflowing_tail_within_the_cap() -> None:
    values = torch.tensor([1.0, 1e-200], dtype=torch.float64)
    decision = select_bond_rank_from_singular_values(
        values,
        threshold=0.0,
        min_bond_rank=1,
        max_bond_rank=12,
    )
    assert decision.threshold_required_rank == 2
    assert decision.retained_rank == 2
    assert decision.zero_spectrum is False
    assert decision.cap_limited is False
    capped = select_bond_rank_from_singular_values(
        values,
        threshold=0.0,
        min_bond_rank=1,
        max_bond_rank=1,
    )
    assert capped.threshold_required_rank == 2
    assert capped.retained_rank == 1
    assert capped.cap_limited is True
    assert capped.relative_discarded_weight == 0.0
    assert capped.threshold_satisfied is False


def test_non_finite_singular_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="NaN または Inf"):
        select_bond_rank_from_singular_values(
            torch.tensor([1.0, float("nan")]),
            threshold=1e-4,
            min_bond_rank=1,
            max_bond_rank=12,
        )
    with pytest.raises(ValueError, match="NaN または Inf"):
        select_bond_rank_from_singular_values(
            torch.tensor([1.0, float("inf")]),
            threshold=1e-4,
            min_bond_rank=1,
            max_bond_rank=12,
        )


def test_available_rank_below_the_cap_is_the_shape_limit() -> None:
    decision = select_bond_rank_from_singular_values(
        torch.tensor([1.0, 0.2, 0.1]),
        threshold=0.0,
        min_bond_rank=1,
        max_bond_rank=12,
    )
    assert decision.available_rank == 3
    assert decision.retained_rank == 3
    assert decision.cap_limited is False
    assert decision.threshold_satisfied is True


def test_delta_zero_is_not_a_fixed_rank_mode() -> None:
    torch.manual_seed(0)
    merged = torch.zeros(1, 2, 2, 2, 2, 1)
    merged[0, 0, 0, 0, 0, 0] = 1.0
    fixed = split_merged_tt_matrix(merged, max_bond=4, direction="right")
    adaptive, decision = split_merged_tt_matrix_by_discarded_weight(
        merged,
        threshold=0.0,
        min_bond_rank=1,
        max_bond_rank=4,
        direction="right",
    )
    assert fixed.retained_rank == 4
    assert decision.retained_rank == 1
    assert adaptive.retained_rank == 1
    assert decision.zero_spectrum is False


def test_fixed_rank_split_matches_the_existing_max_bond_path() -> None:
    torch.manual_seed(1)
    merged = torch.randn(2, 3, 2, 2, 3, 2, dtype=torch.float64)
    for direction in ("right", "left"):
        fixed = split_merged_tt_matrix(merged, max_bond=12, direction=direction)
        full = split_merged_tt_matrix(merged, max_bond=None, direction=direction)
        assert fixed.retained_rank == full.retained_rank
        reconstructed = torch.tensordot(fixed.left_core, fixed.right_core, dims=([3], [0]))
        assert torch.allclose(reconstructed, merged, rtol=1e-6, atol=1e-6)


def test_rank_can_increase_after_an_earlier_truncation() -> None:
    left = torch.zeros(2, 2, 2, 1)
    right = torch.zeros(1, 2, 2, 2)
    left[0, 0, 0, 0] = 1.0
    right[0, 0, 0, 0] = 1.0
    merged = merge_tt_matrix_cores(left, right)
    low, low_decision = split_merged_tt_matrix_by_discarded_weight(
        merged,
        threshold=1.0,
        min_bond_rank=1,
        max_bond_rank=12,
    )
    assert low_decision.retained_rank == 1
    perturbed = merged + 0.05 * torch.randn_like(merged)
    increased, increased_decision = split_merged_tt_matrix_by_discarded_weight(
        perturbed,
        threshold=1e-8,
        min_bond_rank=1,
        max_bond_rank=12,
    )
    assert increased_decision.retained_rank > low.retained_rank
    assert increased.left_core.shape[3] == increased_decision.retained_rank


def _canonical_errors(cores, out_modes, in_modes, center: int) -> list[float]:
    errors = tt_matrix_canonicality_errors(cores, out_modes, in_modes, center)
    return [float(value) for value in errors["left"] + errors["right"]]


def test_directional_split_moves_the_center_and_keeps_outer_isometry() -> None:
    torch.manual_seed(2)
    out_modes = (2, 2, 2)
    in_modes = (2, 2, 2)
    cores = canonicalize_tt_matrix_cores(
        dense_to_tt_matrix_cores(torch.randn(8, 8), out_modes, in_modes, max_rank=4),
        out_modes,
        in_modes,
        0,
    )
    merged = merge_tt_matrix_cores(cores[0], cores[1])
    split, _decision = split_merged_tt_matrix_by_discarded_weight(
        merged,
        threshold=1e-2,
        min_bond_rank=1,
        max_bond_rank=4,
        direction="right",
    )
    updated = replace_tt_matrix_core_pair(cores, 0, split.left_core, split.right_core)
    assert max(_canonical_errors(updated, out_modes, in_modes, 1)) < 1e-5
    original = tt_matrix_to_dense(cores, out_modes, in_modes)
    reconstructed = tt_matrix_to_dense(updated, out_modes, in_modes)
    global_squared = float((reconstructed - original).square().sum())
    local_squared = float(split.discarded_squared_norm)
    assert global_squared == pytest.approx(local_squared, rel=1e-4, abs=1e-6)

    left_centered = canonicalize_tt_matrix_cores(updated, out_modes, in_modes, 2)
    merged_left = merge_tt_matrix_cores(left_centered[1], left_centered[2])
    left_split, _left_decision = split_merged_tt_matrix_by_discarded_weight(
        merged_left,
        threshold=1e-2,
        min_bond_rank=1,
        max_bond_rank=4,
        direction="left",
    )
    moved = replace_tt_matrix_core_pair(
        left_centered,
        1,
        left_split.left_core,
        left_split.right_core,
    )
    assert max(_canonical_errors(moved, out_modes, in_modes, 1)) < 1e-5


def _tt_mlp(max_rank: int) -> MNISTMLP:
    model = MNISTMLP()
    cores = dense_to_tt_matrix_cores(
        model.fc1.weight.detach(),
        OUT_MODES,
        IN_MODES,
        max_rank=max_rank,
    )
    bias = model.fc1.bias.detach().clone()
    model.fc1 = TTLinear(cores, OUT_MODES, IN_MODES, bias=bias)
    canonical = canonicalize_tt_matrix_cores(list(model.fc1.cores), OUT_MODES, IN_MODES, 0)
    assign_tt_linear_cores(model.fc1, canonical, requires_grad=False)
    return model


def test_fixed_rank_update_does_not_use_the_threshold_decision() -> None:
    torch.manual_seed(3)
    model = _tt_mlp(4)
    candidate = copy.deepcopy(model)
    images = torch.randn(4, 1, 28, 28)
    labels = torch.randint(0, 10, (4,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=2)
    before = [core.data_ptr() for core in model.fc1.cores]
    update = update_tt_linear_two_core_pair(
        model,
        candidate,
        loader,
        layer_name="fc1",
        pair_index=0,
        direction="right",
        center=0,
        rank_cap=2,
        tt_pair_learning_rate=0.3,
        dense_parameter_learning_rate=0.3,
        diagnostic_images=images[:2],
        diagnostic_labels=labels[:2],
        device=torch.device("cpu"),
        batch_size=None,
    )
    assert update.bond_rank_decision is None
    assert update.retained_rank <= 2
    assert update.center == 1
    assert tt_matrix_ranks(list(model.fc1.cores))[1] == update.retained_rank
    assert model.fc1.cores[0].data_ptr() != before[0] or model.fc1.cores[0].shape[-1] == update.retained_rank


def test_variable_shape_checkpoint_restores_forward_and_drops_old_storage() -> None:
    torch.manual_seed(4)
    model = _tt_mlp(3)
    images = torch.randn(2, 1, 28, 28)
    model.eval()
    with torch.no_grad():
        expected = model(images).detach().clone()
    payload = {
        "state_dict": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "out_modes": list(OUT_MODES),
        "in_modes": list(IN_MODES),
    }
    old_parameter = model.fc1.cores[1]
    restored = MNISTMLP()
    state = payload["state_dict"]
    cores = [state[f"fc1.cores.{index}"] for index in range(len(OUT_MODES))]
    restored.fc1 = TTLinear(cores, OUT_MODES, IN_MODES, bias=state["fc1.bias"])
    restored.load_state_dict(state)
    restored.eval()
    with torch.no_grad():
        actual = restored(images)
    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert tuple(tt_matrix_ranks(list(restored.fc1.cores))) == tuple(tt_matrix_ranks(cores))
    narrower = [core.clone() for core in restored.fc1.cores]
    bond = narrower[1].shape[0]
    if bond > 1:
        narrower[0] = narrower[0][..., :1].contiguous()
        narrower[1] = narrower[1][:1].contiguous()
        assign_tt_linear_cores(restored.fc1, narrower, requires_grad=False)
        assert old_parameter.data_ptr() not in {parameter.data_ptr() for parameter in restored.parameters()}
        assert restored.fc1.cores[1].shape[0] == 1


def test_resume_from_the_new_center_uses_the_saved_shapes() -> None:
    torch.manual_seed(5)
    model = _tt_mlp(3)
    candidate = copy.deepcopy(model)
    images = torch.randn(4, 1, 28, 28)
    labels = torch.randint(0, 10, (4,))
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)
    first = update_tt_linear_two_core_pair(
        model,
        candidate,
        loader,
        layer_name="fc1",
        pair_index=0,
        direction="right",
        center=0,
        rank_cap=3,
        tt_pair_learning_rate=0.3,
        dense_parameter_learning_rate=0.3,
        diagnostic_images=images[:2],
        diagnostic_labels=labels[:2],
        device=torch.device("cpu"),
        discarded_weight_threshold=1e-2,
        min_bond_rank=1,
    )
    saved_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    saved_center = first.center
    resumed = MNISTMLP()
    cores = [saved_state[f"fc1.cores.{index}"] for index in range(len(OUT_MODES))]
    resumed.fc1 = TTLinear(cores, OUT_MODES, IN_MODES, bias=saved_state["fc1.bias"])
    resumed.load_state_dict(saved_state)
    resumed_candidate = copy.deepcopy(resumed)
    second = update_tt_linear_two_core_pair(
        resumed,
        resumed_candidate,
        loader,
        layer_name="fc1",
        pair_index=1,
        direction="right",
        center=saved_center,
        rank_cap=12,
        tt_pair_learning_rate=0.3,
        dense_parameter_learning_rate=0.3,
        diagnostic_images=images[:2],
        diagnostic_labels=labels[:2],
        device=torch.device("cpu"),
        discarded_weight_threshold=1e-3,
        min_bond_rank=1,
    )
    assert saved_center == 1
    assert second.center == 2
    assert second.bond_rank_decision is not None
    assert tuple(tt_matrix_ranks(list(resumed.fc1.cores)))[2] == second.retained_rank


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU常駐経路は CUDA が必要です。")
def test_gpu_resident_adaptive_update_runs_one_step() -> None:
    torch.manual_seed(6)
    device = torch.device("cuda")
    model = _tt_mlp(3).to(device)
    candidate = copy.deepcopy(model)
    images = torch.randn(4, 1, 28, 28, device=device)
    labels = torch.randint(0, 10, (4,), device=device)
    update = update_tt_linear_two_core_pair(
        model,
        candidate,
        GpuResidentTensorBatches(images, labels),
        layer_name="fc1",
        pair_index=0,
        direction="right",
        center=0,
        rank_cap=12,
        tt_pair_learning_rate=0.3,
        dense_parameter_learning_rate=0.3,
        diagnostic_images=images[:2],
        diagnostic_labels=labels[:2],
        device=device,
        batch_size=2,
        discarded_weight_threshold=1e-4,
        min_bond_rank=1,
    )
    assert update.center == 1
    assert update.bond_rank_decision is not None
    assert torch.isfinite(torch.tensor(update.diagnostic_batch_loss))
