"""1往復ずつの呼び出しが、連続した2往復の final と一致するかを合成データで確認する。"""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nn_compression.compression import factorize_named_tt_linear
from nn_compression.training.tt_linear_scoped_update import (
    TTUpdateScope,
    train_scoped_full_batch_budget,
)

OUT_MODES = (2, 2, 2)
IN_MODES = (2, 2, 2)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8, dtype=torch.float64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 3, dtype=torch.float64)

    def forward(self, x):
        return self.fc2(self.relu1(self.fc1(x)))


def _run(model, features, labels, validation_features, validation_labels, sweeps):
    interval = 2 * (len(model.fc1.cores) - 1) * 1
    train_loader = DataLoader(TensorDataset(features, labels), batch_size=4, shuffle=False)
    validation_loader = DataLoader(
        TensorDataset(validation_features, validation_labels), batch_size=4, shuffle=False,
    )
    return train_scoped_full_batch_budget(
        model,
        train_loader,
        train_loader,
        validation_loader,
        scope=TTUpdateScope("joint"),
        layer_name="fc1",
        batch_size=4,
        total_updates=sweeps * interval,
        evaluation_interval=interval,
        rank_cap=4,
        tt_learning_rate=0.05,
        dense_learning_rate=0.05,
        diagnostic_images=features[:2],
        diagnostic_labels=labels[:2],
        device=torch.device("cpu"),
        local_steps_per_pair=1,
        max_sweeps=sweeps,
        center=0,
    )


def test_split_round_trips_match_continuous_final():
    torch.manual_seed(0)
    model = factorize_named_tt_linear(TinyMLP(), "fc1", OUT_MODES, IN_MODES, max_rank=4)
    generator = torch.Generator().manual_seed(1)
    features = torch.randn(7, 8, generator=generator, dtype=torch.float64)
    labels = torch.randint(0, 3, (7,), generator=generator)
    validation_features = torch.randn(5, 8, generator=generator, dtype=torch.float64)
    validation_labels = torch.randint(0, 3, (5,), generator=generator)

    continuous = copy.deepcopy(model)
    split = copy.deepcopy(model)
    continuous_result = _run(continuous, features, labels, validation_features, validation_labels, 2)
    first = _run(split, features, labels, validation_features, validation_labels, 1)
    second = _run(split, features, labels, validation_features, validation_labels, 1)

    assert first.center == 0
    assert second.center == 0
    assert len(continuous_result.history) == 3
    assert len(first.history) == 2
    assert len(second.history) == 2
    assert first.history[1]["early_stopping_validation_loss"] == second.history[0]["early_stopping_validation_loss"]
    assert continuous_result.history[1]["early_stopping_validation_loss"] == first.history[1]["early_stopping_validation_loss"]
    assert continuous_result.history[2]["early_stopping_validation_loss"] == second.history[1]["early_stopping_validation_loss"]

    with torch.no_grad():
        continuous_output = continuous(features)
        split_output = split(features)
    assert torch.allclose(continuous_output, split_output, rtol=0.0, atol=1e-8)
    for name, parameter in continuous.named_parameters():
        other = dict(split.named_parameters())[name]
        assert torch.allclose(parameter, other, rtol=0.0, atol=1e-8)

    if continuous_result.best_update_count != continuous_result.completed_updates:
        best = continuous_result.best_state
        assert any(
            not torch.equal(parameter.detach().cpu(), best[name].detach().cpu())
            for name, parameter in continuous.named_parameters()
        )
