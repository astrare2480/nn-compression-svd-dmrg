"""parameter snapshot の保存と一括復元。"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nn_compression.utils import restore_named_parameters, snapshot_named_parameters


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.second = nn.Linear(2, 1)
        self.first = nn.Linear(3, 2)

    def forward(self, x):
        return self.second(self.first(x))


def test_snapshot_is_a_cpu_clone_independent_of_later_changes():
    model = Tiny()
    snapshot = snapshot_named_parameters(model)
    assert set(snapshot) == {name for name, _parameter in model.named_parameters()}
    with torch.no_grad():
        model.first.weight.add_(1)
    assert not torch.equal(snapshot["first.weight"], model.first.weight.cpu())
    assert snapshot["first.weight"].device.type == "cpu"
    snapshot["first.weight"].add_(3)
    restored = snapshot_named_parameters(model)
    assert not torch.equal(restored["first.weight"], snapshot["first.weight"])


def test_restore_ignores_registration_order_and_keeps_device():
    model = Tiny()
    snapshot = snapshot_named_parameters(model)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    device_before = {name: parameter.device for name, parameter in model.named_parameters()}
    reversed_snapshot = {name: snapshot[name] for name in reversed(tuple(snapshot))}
    before = {name: value.clone() for name, value in reversed_snapshot.items()}
    restore_named_parameters(model, reversed_snapshot)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter.detach().cpu(), before[name])
        assert parameter.device == device_before[name]
    for name, value in reversed_snapshot.items():
        assert torch.equal(value, before[name])


def test_swapped_parameter_storage_is_restored_without_changing_an_independent_snapshot():
    class Pair(nn.Module):
        def __init__(self):
            super().__init__()
            self.left = nn.Parameter(torch.tensor([1.0, 2.0]))
            self.right = nn.Parameter(torch.tensor([3.0, 4.0]))

    model = Pair()
    original_left = model.left.detach().clone()
    original_right = model.right.detach().clone()
    swapped = {"left": model.right, "right": model.left}
    restore_named_parameters(model, swapped)
    assert torch.equal(model.left, original_right)
    assert torch.equal(model.right, original_left)
    assert model.left.device.type == "cpu"
    assert model.right.device.type == "cpu"

    fresh = Pair()
    with torch.no_grad():
        fresh.left.zero_()
        fresh.right.zero_()
    buffer = torch.tensor([1.0, 2.0, 3.0, 4.0])
    shared = {"left": buffer[:2], "right": buffer[2:]}
    before = buffer.clone()
    device_before = {name: parameter.device for name, parameter in fresh.named_parameters()}
    restore_named_parameters(fresh, shared)
    assert torch.equal(buffer, before)
    assert torch.equal(fresh.left.detach().cpu(), before[:2])
    assert torch.equal(fresh.right.detach().cpu(), before[2:])
    for name, parameter in fresh.named_parameters():
        assert parameter.device == device_before[name]


def test_mismatch_restores_nothing():
    model = Tiny()
    snapshot = snapshot_named_parameters(model)
    with torch.no_grad():
        model.first.weight.fill_(4)
        model.second.weight.fill_(5)
    changed = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    broken = dict(snapshot)
    broken["second.weight"] = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="shape"):
        restore_named_parameters(model, broken)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, changed[name])
    wrong_dtype = dict(snapshot)
    wrong_dtype["first.bias"] = snapshot["first.bias"].to(torch.float64)
    with pytest.raises(ValueError, match="dtype"):
        restore_named_parameters(model, wrong_dtype)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, changed[name])
