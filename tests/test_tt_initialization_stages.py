"""TT 初期化の A/B/C と、更新しない勾配尺度のテスト。"""

import copy

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from nn_compression.compression import tt_linear_initialization_stages
from nn_compression.training import measure_tt_pair_update_scales
from nn_compression.tt_mps import tt_matrix_to_dense

OUT_MODES = (2, 2, 2)
IN_MODES = (2, 2, 2)


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8, dtype=torch.float64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(8, 3, dtype=torch.float64)
        self.register_buffer("marker", torch.tensor([3.0, 4.0], dtype=torch.float64))

    def forward(self, x):
        return self.fc2(self.relu1(self.fc1(x)))


def test_initialization_stages_are_independent_and_canonicalization_keeps_weight():
    torch.manual_seed(0)
    source = TinyMLP()
    source_weight = source.fc1.weight.detach().clone()
    source_ptr = source.fc1.weight.data_ptr()
    stage_a, stage_b, stage_c = tt_linear_initialization_stages(
        source,
        "fc1",
        OUT_MODES,
        IN_MODES,
        max_rank=2,
        canonical_center=0,
    )

    assert source.fc1.weight.data_ptr() == source_ptr
    assert torch.equal(source.fc1.weight, source_weight)
    assert stage_a.fc1.weight.data_ptr() != source_ptr
    assert stage_b.fc2.weight.data_ptr() != stage_c.fc2.weight.data_ptr()
    assert stage_b.fc1.cores[0].data_ptr() != stage_c.fc1.cores[0].data_ptr()

    for left, right in ((stage_a, stage_b), (stage_b, stage_c)):
        assert torch.equal(left.fc1.bias, right.fc1.bias)
        assert torch.equal(left.fc2.weight, right.fc2.weight)
        assert torch.equal(left.fc2.bias, right.fc2.bias)
        assert torch.equal(left.marker, right.marker)
        assert left.fc2.weight.dtype == torch.float64
        assert left.fc2.weight.device == right.fc2.weight.device

    weight_b = tt_matrix_to_dense(list(stage_b.fc1.cores), OUT_MODES, IN_MODES)
    weight_c = tt_matrix_to_dense(list(stage_c.fc1.cores), OUT_MODES, IN_MODES)
    # float64 の QR 正準化。6 要素ではなく 8×8 なので atol=1e-8 は丸めを見られる桁。
    torch.testing.assert_close(weight_c, weight_b, rtol=1e-8, atol=1e-8)
    assert weight_b.shape == source.fc1.weight.shape
    assert weight_b.dtype == torch.float64

    stage_c.fc1.cores[0].data.add_(1.0)
    weight_b_after = tt_matrix_to_dense(list(stage_b.fc1.cores), OUT_MODES, IN_MODES)
    torch.testing.assert_close(weight_b_after, weight_b)


def test_update_scale_measurement_does_not_change_parameters():
    torch.manual_seed(1)
    _source = TinyMLP()
    _a, _b, stage_c = tt_linear_initialization_stages(
        _source, "fc1", OUT_MODES, IN_MODES, max_rank=2, canonical_center=0
    )
    features = torch.randn(5, 8, dtype=torch.float64)
    labels = torch.randint(0, 3, (5,))
    loader = DataLoader(TensorDataset(features, labels), batch_size=2, shuffle=False)
    before = copy.deepcopy(stage_c.state_dict())
    requires_grad = {
        name: parameter.requires_grad for name, parameter in stage_c.named_parameters()
    }
    scales = measure_tt_pair_update_scales(
        stage_c,
        loader,
        layer_name="fc1",
        pair_index=0,
        tt_pair_learning_rate=1e-3,
        dense_parameter_learning_rate=1e-3,
        device=torch.device("cpu"),
    )
    for key, value in stage_c.state_dict().items():
        assert torch.equal(value, before[key])
    for name, parameter in stage_c.named_parameters():
        assert parameter.requires_grad is requires_grad[name]
    assert scales["pair_index"] == 0
    assert scales["num_samples"] == 5
    assert scales["tt_parameter_norm"] > 0
    assert scales["tt_gradient_norm"] > 0
    assert scales["dense_gradient_norm"] > 0
    assert scales["tt_relative_update"] == pytest.approx(
        1e-3 * scales["tt_gradient_norm"] / scales["tt_parameter_norm"]
    )
    assert scales["gradients_finite"] is True
