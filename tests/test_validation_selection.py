"""strict best と stopping reference の分離。"""

import csv
import math
from pathlib import Path

import pytest

from nn_compression.training import (
    initial_validation_selection,
    replay_validation_selection,
    update_validation_selection,
)


ROOT = Path(__file__).resolve().parents[1]
JOINT_HISTORY = (
    ROOT
    / "models"
    / "40_dmrg"
    / "10_fashion_mnist_mlp"
    / "05_update_scope_ablation_gpu_resident"
    / "joint"
    / "history.csv"
)


def test_small_improvement_updates_strict_best_but_not_reference():
    state = initial_validation_selection(1.0, 0)
    state = update_validation_selection(
        state, 0.99995, 1, min_delta=1e-4, patience=5
    )
    assert state.strict_best_loss == pytest.approx(0.99995)
    assert state.strict_best_step == 1
    assert state.reference_loss == pytest.approx(1.0)
    assert state.non_improvement_count == 1
    assert state.early_stop_triggered is False


def test_cumulative_improvement_moves_reference_without_resetting_each_tiny_step():
    state = initial_validation_selection(1.0, 0)
    state = update_validation_selection(
        state, 0.99995, 1, min_delta=1e-4, patience=5
    )
    state = update_validation_selection(
        state, 0.9998, 2, min_delta=1e-4, patience=5
    )
    assert state.reference_loss == pytest.approx(0.9998)
    assert state.strict_best_step == 2
    assert state.non_improvement_count == 0
    assert state.early_stop_triggered is False


def test_patience_boundary_and_equal_loss_keeps_earlier_best():
    state = initial_validation_selection(1.0, 0)
    state = update_validation_selection(state, 1.0, 1, min_delta=1e-4, patience=2)
    assert state.strict_best_step == 0
    assert state.non_improvement_count == 1
    assert state.early_stop_triggered is False
    state = update_validation_selection(state, 1.0, 2, min_delta=1e-4, patience=2)
    assert state.early_stop_triggered is True
    assert state.non_improvement_count == 2
    with pytest.raises(RuntimeError):
        update_validation_selection(state, 0.5, 3, min_delta=1e-4, patience=2)


def test_min_delta_zero_equal_loss_stops_by_patience():
    state = initial_validation_selection(1.0, 0)
    state = update_validation_selection(state, 1.0, 1, min_delta=0.0, patience=2)
    assert state.strict_best_loss == pytest.approx(1.0)
    assert state.strict_best_step == 0
    assert state.reference_loss == pytest.approx(1.0)
    assert state.non_improvement_count == 1
    assert state.early_stop_triggered is False
    state = update_validation_selection(state, 1.0, 2, min_delta=0.0, patience=2)
    assert state.reference_loss == pytest.approx(1.0)
    assert state.strict_best_step == 0
    assert state.non_improvement_count == 2
    assert state.early_stop_triggered is True


def test_min_delta_zero_any_decrease_is_an_improvement():
    state = initial_validation_selection(1.0, 0)
    updated = update_validation_selection(
        state, 1.0 - 1e-12, 4, min_delta=0.0, patience=5
    )
    assert updated.strict_best_loss == pytest.approx(1.0 - 1e-12)
    assert updated.strict_best_step == 4
    assert updated.reference_loss == pytest.approx(1.0 - 1e-12)
    assert updated.non_improvement_count == 0
    assert updated.early_stop_triggered is False


def test_positive_min_delta_exact_boundary_still_moves_reference():
    state = initial_validation_selection(1.0, 0)
    just_short = 1.0 - 1e-4
    held = update_validation_selection(
        state, just_short, 1, min_delta=1e-4, patience=5
    )
    assert 1.0 - just_short < 1e-4
    assert held.strict_best_loss == pytest.approx(just_short)
    assert held.strict_best_step == 1
    assert held.reference_loss == pytest.approx(1.0)
    assert held.non_improvement_count == 1
    reached = math.nextafter(just_short, 0.0)
    moved = update_validation_selection(
        state, reached, 2, min_delta=1e-4, patience=5
    )
    assert 1.0 - reached >= 1e-4
    assert moved.reference_loss == pytest.approx(reached)
    assert moved.strict_best_loss == pytest.approx(reached)
    assert moved.strict_best_step == 2
    assert moved.non_improvement_count == 0


def test_non_finite_is_not_a_completed_observation():
    state = initial_validation_selection(1.0, 0)
    with pytest.raises(ValueError):
        update_validation_selection(state, float("nan"), 1, min_delta=1e-4, patience=5)
    with pytest.raises(ValueError):
        initial_validation_selection(float("inf"), 0)


def test_replay_stops_at_patience():
    states = replay_validation_selection(
        [1.0, 1.0, 1.0, 0.0],
        [0, 1, 2, 3],
        min_delta=1e-4,
        patience=2,
    )
    assert len(states) == 3
    assert states[-1].early_stop_triggered is True


def test_notebook05_joint_history_has_no_early_stop_and_zero_stall():
    if not JOINT_HISTORY.is_file():
        pytest.skip("Notebook 05 joint history がありません。")
    with JOINT_HISTORY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    losses = [float(row["early_stopping_validation_loss"]) for row in rows]
    steps = [int(row["cumulative_updates"]) for row in rows]
    states = replay_validation_selection(losses, steps, min_delta=1e-4, patience=5)
    assert len(states) == len(rows)
    assert all(state.early_stop_triggered is False for state in states)
    assert states[-1].non_improvement_count == 0
    assert states[-1].strict_best_step == 3200
    assert states[-1].strict_best_loss == pytest.approx(0.34194087901115416)
    assert float(rows[-1]["early_stopping_validation_accuracy"]) == pytest.approx(0.8716)
