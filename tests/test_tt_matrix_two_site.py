"""TT-matrix の隣接2 core 結合・forward・SVD 再分割のテスト。"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nn_compression.compression import (
    TTLinear,
    assign_tt_linear_cores,
    module_forward_with_merged_tt_linear_pair,
    module_forward_with_tt_linear_cores,
    tt_linear_forward,
    tt_linear_forward_with_merged_pair,
)
from nn_compression.tt_mps import (
    canonicalize_tt_matrix_cores,
    merge_tt_matrix_cores,
    merged_tt_matrix_to_matrix,
    replace_tt_matrix_core_pair,
    split_merged_tt_matrix,
    tt_cores_to_tt_matrix_cores,
    tt_matrix_canonicality_errors,
    tt_matrix_cores_to_tt_cores,
    tt_matrix_to_dense,
)
from nn_compression.training import parameter_change_summary

OUT_MODES = (2, 3, 2)
IN_MODES = (2, 2, 3)
RANKS = (1, 2, 3, 1)


def _cores(seed=0, *, dtype=torch.float64, ranks=RANKS, out_modes=OUT_MODES, in_modes=IN_MODES):
    generator = torch.Generator().manual_seed(seed)
    cores = []
    for k, (m_k, n_k) in enumerate(zip(out_modes, in_modes)):
        cores.append(
            torch.randn(
                ranks[k],
                m_k,
                n_k,
                ranks[k + 1],
                generator=generator,
                dtype=torch.float64,
            ).to(dtype=dtype)
        )
    return cores


def _tolerances(dtype):
    if dtype == torch.float32:
        return {"rtol": 2e-4, "atol": 2e-4}
    return {"rtol": 1e-8, "atol": 1e-8}


def test_matrix_core_reshape_roundtrip_keeps_values_and_graph():
    cores = _cores()
    cores[1].requires_grad_(True)
    restored = tt_cores_to_tt_matrix_cores(
        tt_matrix_cores_to_tt_cores(cores, OUT_MODES, IN_MODES),
        OUT_MODES,
        IN_MODES,
    )
    for original, again in zip(cores, restored):
        torch.testing.assert_close(again, original)
    restored[1].sum().backward()
    assert cores[1].grad is not None
    assert cores[1].grad.shape == cores[1].shape


@pytest.mark.parametrize("pair", [0, 1])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_merged_forward_matches_two_cores_and_keeps_grad_on_merged_only(pair, dtype):
    cores = _cores(dtype=dtype)
    for core in cores:
        core.requires_grad_(True)
    merged = merge_tt_matrix_cores(cores[pair], cores[pair + 1])
    assert merged.shape == (
        cores[pair].shape[0],
        OUT_MODES[pair],
        IN_MODES[pair],
        OUT_MODES[pair + 1],
        IN_MODES[pair + 1],
        cores[pair + 1].shape[-1],
    )
    # 学習変数は元 core のグラフから切り離す。結合関数自体は detach しない。
    merged_leaf = merged.detach().clone().requires_grad_(True)
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(4, 12, generator=generator, dtype=torch.float64).to(dtype=dtype)
    bias = torch.randn(12, generator=generator, dtype=torch.float64).to(dtype=dtype)

    reference = tt_linear_forward(x, cores, OUT_MODES, IN_MODES, bias)
    via_merged = tt_linear_forward_with_merged_pair(
        x, cores, pair, merged_leaf, OUT_MODES, IN_MODES, bias
    )
    torch.testing.assert_close(via_merged, reference, **_tolerances(dtype))

    # pair の値を壊しても、forward は merged だけを使う。
    broken = [core.clone() for core in cores]
    broken[pair] = torch.zeros_like(broken[pair])
    broken[pair + 1] = torch.ones_like(broken[pair + 1])
    via_broken = tt_linear_forward_with_merged_pair(
        x, broken, pair, merged_leaf, OUT_MODES, IN_MODES, bias
    )
    torch.testing.assert_close(via_broken, via_merged)

    via_merged.square().sum().backward()
    assert merged_leaf.grad is not None
    assert merged_leaf.grad.shape == merged_leaf.shape
    assert torch.isfinite(merged_leaf.grad).all()
    for index, core in enumerate(cores):
        if index in (pair, pair + 1):
            assert core.grad is None
        else:
            assert core.grad is not None


def test_merge_keeps_graph_until_the_caller_detaches():
    cores = _cores()
    cores[0].requires_grad_(True)
    merged = merge_tt_matrix_cores(cores[0], cores[1])
    merged.sum().backward()
    assert cores[0].grad is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_svd_restores_merged_tensor_without_renormalizing(dtype):
    cores = _cores(dtype=dtype)
    merged = merge_tt_matrix_cores(cores[0], cores[1])
    merged = merged * 5
    split = split_merged_tt_matrix(merged, max_bond=None)
    restored = merge_tt_matrix_cores(split.left_core, split.right_core)
    torch.testing.assert_close(restored, merged, **_tolerances(dtype))
    assert split.retained_rank == min(
        merged.shape[0] * merged.shape[1] * merged.shape[2],
        merged.shape[3] * merged.shape[4] * merged.shape[5],
    )
    assert split.singular_values.shape == (split.retained_rank,)
    assert torch.count_nonzero(split.discarded_squared_norm) == 0
    # ノルムを1へ戻さない。
    assert abs(torch.linalg.vector_norm(restored).item() - torch.linalg.vector_norm(merged).item()) < (
        1e-3 if dtype == torch.float32 else 1e-8
    )


@pytest.mark.parametrize("direction", ["right", "left"])
def test_directional_split_places_singular_values_without_renormalizing(direction):
    """右向きは Σ を右 core、左向きは Σ を左 core に置く。既定は右向き。"""
    cores = _cores(dtype=torch.float64)
    merged = merge_tt_matrix_cores(cores[0], cores[1]) * 5
    matrix = merged_tt_matrix_to_matrix(merged)
    u_factor, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    split = split_merged_tt_matrix(merged, max_bond=None, direction=direction)
    if direction == "right":
        expected_left = u_factor.reshape_as(split.left_core)
        expected_right = (singular_values.unsqueeze(1) * vh).reshape_as(split.right_core)
        default = split_merged_tt_matrix(merged, max_bond=None)
        torch.testing.assert_close(default.left_core, split.left_core)
        torch.testing.assert_close(default.right_core, split.right_core)
    else:
        expected_left = (u_factor * singular_values.unsqueeze(0)).reshape_as(split.left_core)
        expected_right = vh.reshape_as(split.right_core)
    torch.testing.assert_close(split.left_core, expected_left)
    torch.testing.assert_close(split.right_core, expected_right)
    restored = merge_tt_matrix_cores(split.left_core, split.right_core)
    torch.testing.assert_close(restored, merged, atol=1e-8, rtol=1e-8)
    assert abs(torch.linalg.vector_norm(restored).item() - 5 * torch.linalg.vector_norm(merged / 5).item()) < 1e-6


def test_truncated_frobenius_error_matches_discarded_singular_values():
    cores = _cores(dtype=torch.float64)
    merged = merge_tt_matrix_cores(cores[0], cores[1])
    matrix = merged_tt_matrix_to_matrix(merged)
    split = split_merged_tt_matrix(merged, max_bond=1)
    restored = merge_tt_matrix_cores(split.left_core, split.right_core)
    residual_sq = (matrix - merged_tt_matrix_to_matrix(restored)).square().sum()
    torch.testing.assert_close(residual_sq, split.discarded_squared_norm, rtol=1e-8, atol=1e-8)
    tail_sq = split.singular_values[split.retained_rank :].square().sum()
    torch.testing.assert_close(tail_sq, split.discarded_squared_norm)


def test_canonical_truncation_error_matches_weight_error_and_gauge_preserves_weight():
    cores = _cores(seed=2)
    weight = tt_matrix_to_dense(cores, OUT_MODES, IN_MODES)
    canonical = canonicalize_tt_matrix_cores(cores, OUT_MODES, IN_MODES, center=0)
    torch.testing.assert_close(
        tt_matrix_to_dense(canonical, OUT_MODES, IN_MODES),
        weight,
        rtol=1e-8,
        atol=1e-8,
    )
    errors = tt_matrix_canonicality_errors(canonical, OUT_MODES, IN_MODES, center=0)
    assert errors["left"] == []
    assert len(errors["right"]) == 2
    for error in errors["right"]:
        torch.testing.assert_close(error, torch.zeros_like(error), atol=1e-8, rtol=0)

    merged = merge_tt_matrix_cores(canonical[0], canonical[1])
    split = split_merged_tt_matrix(merged, max_bond=1)
    replaced = replace_tt_matrix_core_pair(canonical, 0, split.left_core, split.right_core)
    weight_error = torch.linalg.vector_norm(
        tt_matrix_to_dense(replaced, OUT_MODES, IN_MODES) - weight
    )
    local_error = split.discarded_squared_norm.sqrt()
    torch.testing.assert_close(weight_error, local_error, rtol=1e-6, atol=1e-8)


def test_without_canonical_form_local_error_need_not_equal_weight_error():
    """正準化していないとき、局所 Frobenius 誤差を全体の重み誤差と同一視しない。"""
    cores = _cores(seed=2)
    weight = tt_matrix_to_dense(cores, OUT_MODES, IN_MODES)
    merged = merge_tt_matrix_cores(cores[0], cores[1])
    split = split_merged_tt_matrix(merged, max_bond=1)
    replaced = replace_tt_matrix_core_pair(cores, 0, split.left_core, split.right_core)
    weight_error = torch.linalg.vector_norm(
        tt_matrix_to_dense(replaced, OUT_MODES, IN_MODES) - weight
    )
    local_error = split.discarded_squared_norm.sqrt()
    assert abs(weight_error.item() - local_error.item()) > 1e-3


def test_module_forward_uses_merged_tensor_and_restores_the_layer():
    cores = _cores()
    layer = TTLinear(cores, OUT_MODES, IN_MODES, bias=torch.zeros(12, dtype=torch.float64))

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = layer
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(12, 3, dtype=torch.float64)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = Tiny()
    x = torch.randn(5, 12, dtype=torch.float64)
    merged = merge_tt_matrix_cores(cores[0], cores[1]).detach().requires_grad_(True)
    output = module_forward_with_merged_tt_linear_pair(model, x, "fc1", 0, merged)
    manual = model.fc2(
        model.relu(
            tt_linear_forward_with_merged_pair(
                x, list(model.fc1.cores), 0, merged, OUT_MODES, IN_MODES, model.fc1.bias
            )
        )
    )
    torch.testing.assert_close(output, manual)
    output.square().sum().backward()
    assert merged.grad is not None

    split = split_merged_tt_matrix(merged.detach(), max_bond=None)
    replaced = replace_tt_matrix_core_pair(list(model.fc1.cores), 0, split.left_core, split.right_core)
    with torch.no_grad():
        from_cores = module_forward_with_tt_linear_cores(model, x, "fc1", replaced)
    torch.testing.assert_close(from_cores, output.detach(), rtol=1e-8, atol=1e-8)

    # forward の差し替えは呼び出し後に戻る。
    direct = model(x)
    reference = model.fc2(model.relu(tt_linear_forward(x, list(model.fc1.cores), OUT_MODES, IN_MODES, model.fc1.bias)))
    torch.testing.assert_close(direct, reference)


def test_assign_tt_linear_cores_keeps_bias_and_can_change_bond():
    from nn_compression.compression import assign_tt_linear_cores

    cores = _cores()
    bias = torch.arange(12, dtype=torch.float64)
    layer = TTLinear(cores, OUT_MODES, IN_MODES, bias=bias)
    before_bias = layer.bias.detach().clone()
    canonical = canonicalize_tt_matrix_cores(list(layer.cores), OUT_MODES, IN_MODES, center=1)
    assign_tt_linear_cores(layer, canonical, requires_grad=False)
    torch.testing.assert_close(layer.bias, before_bias)
    assert all(not core.requires_grad for core in layer.cores)
    torch.testing.assert_close(
        tt_matrix_to_dense(list(layer.cores), OUT_MODES, IN_MODES),
        tt_matrix_to_dense(cores, OUT_MODES, IN_MODES),
        rtol=1e-8,
        atol=1e-8,
    )

    low_rank = [core.clone() for core in canonical]
    # pair 0 の bond を 1 に落とす。隣との接続は replace の検証に任せる。
    split = split_merged_tt_matrix(merge_tt_matrix_cores(low_rank[0], low_rank[1]), max_bond=1)
    low_rank = replace_tt_matrix_core_pair(low_rank, 0, split.left_core, split.right_core)
    assign_tt_linear_cores(layer, low_rank)
    assert tuple(layer.cores[0].shape)[-1] == 1
    assert tuple(layer.cores[1].shape)[0] == 1
    x = torch.randn(2, 12, dtype=torch.float64)
    y = layer(x)
    assert y.shape == (2, 12)
    assert torch.isfinite(y).all()


def _tiny_model(cores, bias):
    layer = TTLinear(cores, OUT_MODES, IN_MODES, bias=bias)

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = layer
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(12, 3, dtype=cores[0].dtype)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    return Tiny()


def test_assign_tt_linear_cores_writeback_matches_model_forward():
    """書き戻し後の通常 forward が、一時 core 列の forward と一致する。

    同じ shape の copy_ と、bond が変わって Parameter を差し替える場合の両方を見る。
    bias と対象外の層は変えない。shape が変わるとき、既存 optimizer は古い Parameter を
    参照し続ける（state の移行はしない）。
    """
    cores = _cores()
    bias = torch.arange(12, dtype=torch.float64)
    model = _tiny_model(cores, bias)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    x = torch.randn(4, 12, dtype=torch.float64)

    same_shape = [core.detach().clone() for core in model.fc1.cores]
    same_shape[0] = same_shape[0] + 0.1
    bias_before = model.fc1.bias.detach().clone()
    fc2_weight_before = model.fc2.weight.detach().clone()
    fc2_bias_before = model.fc2.bias.detach().clone()
    other_before = model.fc1.cores[2].detach().clone()
    same_parameter = model.fc1.cores[0]
    assign_tt_linear_cores(model.fc1, same_shape)
    assert model.fc1.cores[0] is same_parameter
    assert torch.equal(model.fc1.cores[0], same_shape[0])
    assert torch.equal(model.fc1.cores[1], same_shape[1])
    assert torch.equal(model.fc1.cores[2], other_before)
    assert torch.equal(model.fc1.bias, bias_before)
    assert torch.equal(model.fc2.weight, fc2_weight_before)
    assert torch.equal(model.fc2.bias, fc2_bias_before)
    assert not model.fc2.weight.requires_grad
    assert not model.fc1.bias.requires_grad
    with torch.no_grad():
        temporary = module_forward_with_tt_linear_cores(model, x, "fc1", same_shape)
        direct = model(x)
    torch.testing.assert_close(direct, temporary)
    layer_output = model.fc1(x)
    reference = tt_linear_forward(x, list(model.fc1.cores), OUT_MODES, IN_MODES, model.fc1.bias)
    torch.testing.assert_close(layer_output, reference)
    assert torch.isfinite(layer_output).all()
    assert torch.isfinite(direct).all()

    split = split_merged_tt_matrix(
        merge_tt_matrix_cores(same_shape[0], same_shape[1]),
        max_bond=1,
    )
    low_rank = replace_tt_matrix_core_pair(same_shape, 0, split.left_core, split.right_core)
    assert tuple(low_rank[0].shape) != tuple(model.fc1.cores[0].shape)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    old_core = model.fc1.cores[0]
    bias_before = model.fc1.bias.detach().clone()
    fc2_weight_before = model.fc2.weight.detach().clone()
    other_before = model.fc1.cores[2].detach().clone()
    assign_tt_linear_cores(model.fc1, low_rank)
    assert model.fc1.cores[0] is not old_core
    assert tuple(model.fc1.cores[0].shape)[-1] == 1
    assert tuple(model.fc1.cores[1].shape)[0] == 1
    assert torch.equal(model.fc1.cores[0], low_rank[0])
    assert torch.equal(model.fc1.cores[1], low_rank[1])
    assert torch.equal(model.fc1.cores[2], other_before)
    assert torch.equal(model.fc1.bias, bias_before)
    assert torch.equal(model.fc2.weight, fc2_weight_before)
    assert not model.fc2.weight.requires_grad
    held = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert any(parameter is old_core for parameter in held)
    assert all(parameter is not model.fc1.cores[0] for parameter in held)
    with torch.no_grad():
        temporary = module_forward_with_tt_linear_cores(model, x, "fc1", low_rank)
        direct = model(x)
    torch.testing.assert_close(direct, temporary)
    layer_output = model.fc1(x)
    assert layer_output.shape == (4, 12)
    assert torch.isfinite(layer_output).all()
    assert torch.isfinite(direct).all()


def test_parameter_summary_sees_no_change_when_only_an_external_tensor_is_updated():
    cores = _cores(dtype=torch.float32)
    layer = TTLinear(cores, OUT_MODES, IN_MODES, bias=torch.zeros(12))

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = layer

        def forward(self, x):
            return self.fc1(x)

    model = Tiny()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = copy.deepcopy(model.state_dict())
    merged = merge_tt_matrix_cores(cores[0], cores[1]).detach().clone().requires_grad_(True)
    x = torch.randn(3, 12)
    labels = torch.tensor([0, 1, 0])
    logits = module_forward_with_merged_tt_linear_pair(model, x, "fc1", 0, merged)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    updated = (merged - 1e-2 * merged.grad).detach()
    assert updated.shape == merged.shape
    assert torch.isfinite(updated).all()
    rows = parameter_change_summary(before, model)
    assert rows
    assert all(row["exactly_equal"] for row in rows)
