"""名前指定の中間出力と段階別誤差。CPU の小さいモデルだけを使う。"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nn_compression.metrics import (
    CapturedModuleOutputs,
    StageChangeSummary,
    capture_internal_outputs,
    stage_change_summary,
)


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 4)
        self.act = nn.ReLU()
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        return self.head(self.act(self.hidden(x)))


def test_capture_uses_given_module_names_and_restores_training_flags():
    torch.manual_seed(0)
    model = TinyNet()
    model.train()
    model.act.eval()
    inputs = torch.randn(5, 3)
    captured = capture_internal_outputs(model, inputs, ("hidden", "act"))
    assert isinstance(captured, CapturedModuleOutputs)
    assert captured.output.shape == (5, 2)
    assert captured.module("hidden").shape == (5, 4)
    assert torch.equal(captured.module("act"), torch.relu(captured.module("hidden")))
    assert model.training
    assert model.hidden.training
    assert not model.act.training
    assert model.head.training
    assert model.hidden._forward_hooks == {}
    assert model.act._forward_hooks == {}
    with torch.no_grad():
        expected = model.eval()(inputs)
    model.train()
    model.act.eval()
    assert torch.allclose(captured.output, expected)


def test_hooks_and_training_flags_are_cleared_when_forward_fails():
    class Boom(nn.Module):
        def __init__(self):
            super().__init__()
            self.hidden = nn.Linear(2, 2)

        def forward(self, x):
            self.hidden(x)
            raise RuntimeError("stop")

    model = Boom()
    model.train()
    with pytest.raises(RuntimeError, match="stop"):
        capture_internal_outputs(model, torch.ones(2, 2), ("hidden",))
    assert model.training
    assert model.hidden.training
    assert model.hidden._forward_hooks == {}


def test_linear_output_survives_following_inplace_relu():
    torch.manual_seed(1)
    model = TinyNet()
    model.train()
    model.act = nn.ReLU(inplace=True)
    model.act.eval()
    inputs = torch.randn(6, 3)
    inputs_before = inputs.clone()
    with torch.no_grad():
        expected = torch.nn.functional.linear(inputs, model.hidden.weight, model.hidden.bias)
    assert torch.any(expected < 0)
    parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    training = [(module, module.training) for module in model.modules()]
    captured = capture_internal_outputs(model, inputs, ("hidden",))
    assert torch.equal(captured.module("hidden"), expected)
    assert torch.any(captured.module("hidden") < 0)
    assert model.hidden._forward_hooks == {}
    assert model.act._forward_hooks == {}
    for module, flag in training:
        assert module.training is flag
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, parameters[name])
    assert torch.equal(inputs, inputs_before)


def test_inplace_module_is_rejected_before_forward():
    model = TinyNet()
    model.act = nn.ReLU(inplace=True)
    seen = []
    original = model.forward

    def watch(x):
        seen.append(x)
        return original(x)

    model.forward = watch
    with pytest.raises(ValueError, match="inplace"):
        capture_internal_outputs(model, torch.ones(2, 3), ("act",))
    assert seen == []


def test_stage_change_summary_relative_l2_and_zero_baseline():
    baseline = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    candidate = torch.zeros_like(baseline)
    summary = stage_change_summary(candidate, baseline, "hidden")
    assert isinstance(summary, StageChangeSummary)
    assert summary.stage_name == "hidden"
    assert summary.dataset_relative_l2 == pytest.approx(1.0)
    assert summary.diff_l2_mean == pytest.approx(3.5)
    assert summary.diff_l2_max == pytest.approx(4.0)
    assert torch.allclose(summary.diff_l2, torch.tensor([3.0, 4.0], dtype=torch.float64))
    with pytest.raises(ValueError, match="分母がゼロ"):
        stage_change_summary(candidate, torch.zeros_like(baseline), "hidden")
    with pytest.raises(ValueError, match="形状"):
        stage_change_summary(candidate, baseline[:, :1], "hidden")
