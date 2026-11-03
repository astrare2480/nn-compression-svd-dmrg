"""adaptive TTLinear sweep。CPU の小さい合成データだけを使う。"""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nn_compression.compression import factorize_named_tt_linear
from nn_compression.training import (
    AdaptiveTTLinearSweepResult,
    NonFiniteLocalUpdate,
    train_adaptive_tt_linear_full_batch_sweeps,
)
from nn_compression.training import tt_linear_adaptive_sweep as adaptive

OUT_MODES = (2, 2, 2)
IN_MODES = (2, 2, 2)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, dtype=torch.float64)
        self.head = nn.Linear(8, 3, dtype=torch.float64)

    def forward(self, x):
        return self.head(torch.relu(self.proj(x)))


def _model():
    torch.manual_seed(0)
    model = factorize_named_tt_linear(TinyMLP(), "proj", OUT_MODES, IN_MODES, max_rank=4)
    for core in model.proj.cores:
        core.requires_grad_(False)
    return model


def _data():
    generator = torch.Generator().manual_seed(1)
    train_x = torch.randn(6, 8, generator=generator, dtype=torch.float64)
    train_y = torch.randint(0, 3, (6,), generator=generator)
    val_x = torch.randn(4, 8, generator=generator, dtype=torch.float64)
    val_y = torch.randint(0, 3, (4,), generator=generator)
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=3, shuffle=False)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=4, shuffle=False)
    return train_loader, val_loader, train_x[:2].clone(), train_y[:2].clone()


def _run(model, **overrides):
    train_loader, val_loader, images, labels = _data()
    kwargs = {
        "layer_name": "proj",
        "discarded_weight_threshold": 0.5,
        "rank_cap": 4,
        "tt_pair_learning_rate": 1e-2,
        "dense_parameter_learning_rate": 1e-3,
        "max_sweeps": 1,
        "local_steps_per_pair": 1,
        "min_delta": 0.0,
        "patience": 5,
        "diagnostic_images": images,
        "diagnostic_labels": labels,
        "device": torch.device("cpu"),
    }
    kwargs.update(overrides)
    return train_adaptive_tt_linear_full_batch_sweeps(
        model,
        train_loader,
        val_loader,
        **kwargs,
    )


def test_one_sweep_records_both_directions_and_ranks():
    result = _run(_model())
    assert isinstance(result, AdaptiveTTLinearSweepResult)
    assert result.stop_reason == "max_sweeps"
    assert result.completed_sweeps == 1
    assert result.total_local_updates == 4
    assert result.center == 0
    assert [item.direction for item in result.local_updates] == ["right", "right", "left", "left"]
    assert [item.pair_index for item in result.local_updates] == [0, 1, 1, 0]
    for item in result.local_updates:
        assert item.rank_before >= 1
        assert item.rank_after >= 1
        assert item.rank_after <= 4
        assert 0.0 <= item.relative_discarded_weight <= 1.0
        assert item.discarded_squared_norm >= 0.0
        assert item.actual_ranks[item.pair_index + 1] == item.rank_after
        assert item.center_after in {item.pair_index, item.pair_index + 1}
    assert [row.sweep for row in result.history] == [0, 1]
    assert set(result.best_state) == set(result.final_state)
    assert all(torch.is_tensor(value) for value in result.final_state.values())


def test_early_stopping_keeps_best_state_independent_of_later_model_changes():
    model = _model()
    result = _run(model, max_sweeps=3, min_delta=1e6, patience=1)
    assert result.stop_reason == "early_stopping"
    assert result.completed_sweeps == 1
    assert len(result.history) == 2
    assert result.history[-1].early_stop_triggered
    snapshot = {key: value.clone() for key, value in result.best_state.items()}
    with torch.no_grad():
        model.head.weight.add_(1.0)
    for key, value in snapshot.items():
        assert torch.equal(result.best_state[key], value)


def test_non_finite_update_returns_without_recording_that_step(monkeypatch):
    def fail(*args, **kwargs):
        raise NonFiniteLocalUpdate("synthetic")

    monkeypatch.setattr(adaptive, "update_tt_linear_two_core_pair", fail)
    result = _run(_model())
    assert result.stop_reason == "non_finite"
    assert result.stop_detail == "synthetic"
    assert result.completed_sweeps == 0
    assert result.local_updates == ()
    assert [row.sweep for row in result.history] == [0]


def test_svd_failure_is_a_stop_result(monkeypatch):
    def fail(*args, **kwargs):
        raise torch.linalg.LinAlgError("svd")

    monkeypatch.setattr(adaptive, "update_tt_linear_two_core_pair", fail)
    result = _run(_model())
    assert result.stop_reason == "svd_failure"
    assert result.local_updates == ()


def test_input_validation_rejects_bad_layer_and_threshold():
    model = _model()
    with pytest.raises(TypeError, match="TTLinear"):
        _run(model, layer_name="head")
    with pytest.raises(ValueError, match="discarded_weight_threshold"):
        _run(model, discarded_weight_threshold=-0.1)
