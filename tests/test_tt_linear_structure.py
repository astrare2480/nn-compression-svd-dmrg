"""metrics.tt_linear_structure の構造メトリクス。"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

from nn_compression.compression import build_tt_linear
from nn_compression.metrics import (
    TTLinearStructureMetrics,
    count_parameters,
    linear_macs,
    tt_linear_direct_contraction_macs,
    tt_linear_structure_metrics,
)
from nn_compression.models import MNISTMLP
from nn_compression.tt_mps import tt_matrix_ranks

OUT_MODES = (4, 2, 2, 2, 4, 4)
IN_MODES = (2, 2, 2, 2, 7, 7)


def _tt_mlp(max_rank=4, layer_name="fc1"):
    torch.manual_seed(0)
    model = MNISTMLP()
    setattr(model, layer_name, build_tt_linear(getattr(model, layer_name), OUT_MODES, IN_MODES, max_rank=max_rank))
    return model


def test_metrics_match_existing_rank_parameter_and_macs_apis():
    model = _tt_mlp(4)
    metrics = tt_linear_structure_metrics(model, "fc1")
    ranks = tuple(int(rank) for rank in tt_matrix_ranks(list(model.fc1.cores)))
    fc1_macs = int(sum(tt_linear_direct_contraction_macs(OUT_MODES, IN_MODES, ranks)))
    assert isinstance(metrics, TTLinearStructureMetrics)
    assert metrics.layer_name == "fc1"
    assert metrics.actual_ranks == ranks
    assert metrics.tt_core_elements == sum(core.numel() for core in model.fc1.cores)
    assert metrics.layer_parameters == sum(
        parameter.numel() for name, parameter in model.named_parameters() if name.startswith("fc1.")
    )
    assert metrics.total_parameters == count_parameters(model)
    assert metrics.layer_macs == fc1_macs
    assert metrics.total_macs == fc1_macs + linear_macs(model.fc2) + linear_macs(model.fc3)


def test_rank12_reference_values_from_notebook06():
    metrics = tt_linear_structure_metrics(_tt_mlp(12), "fc1")
    assert metrics.actual_ranks == (1, 8, 12, 12, 12, 12, 1)
    assert metrics.tt_core_elements == 5_968
    assert metrics.total_parameters == 140_378
    assert metrics.layer_macs == 2_175_488
    assert metrics.total_macs == 2_309_120


def test_layer_name_is_an_argument():
    torch.manual_seed(0)
    model = nn.Sequential()
    model.add_module("head", nn.Linear(10, 784))
    model.add_module("block", nn.Sequential(build_tt_linear(nn.Linear(784, 512), OUT_MODES, IN_MODES, max_rank=2)))
    metrics = tt_linear_structure_metrics(model, "block.0")
    assert metrics.layer_name == "block.0"
    assert metrics.total_macs == metrics.layer_macs + 10 * 784


def test_result_is_frozen_and_model_is_unchanged():
    model = _tt_mlp(3)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    training = model.training
    metrics = tt_linear_structure_metrics(model, "fc1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        metrics.total_macs = 0
    assert model.training == training
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_rejects_dense_layer_bad_name_and_unknown_parametric_modules():
    with pytest.raises(TypeError, match="TTLinear"):
        tt_linear_structure_metrics(MNISTMLP(), "fc1")
    with pytest.raises(AttributeError):
        tt_linear_structure_metrics(_tt_mlp(2), "fc9")
    with pytest.raises(ValueError):
        tt_linear_structure_metrics(_tt_mlp(2), "")
    with pytest.raises(TypeError):
        tt_linear_structure_metrics("model", "fc1")
    model = _tt_mlp(2)
    model.norm = nn.LayerNorm(10)
    with pytest.raises(TypeError, match="norm"):
        tt_linear_structure_metrics(model, "fc1")
