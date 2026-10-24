"""``training.parameter_scope`` のテスト。"""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import torch
from torch import nn

from nn_compression.compression import build_tt_conv2d, build_tt_linear
from nn_compression.training import (
    parameter_change_summary,
    set_trainable_parameter_prefixes,
    trainable_named_parameters,
)


class _TTMLP(nn.Module):
    """``fc1.cores.*`` を持つ TT 層と、prefix 境界の罠になる名前を持つ小さなモデル。"""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        dense = nn.Linear(16, 8, dtype=torch.float64)
        with torch.no_grad():
            dense.weight.copy_(torch.randn(8, 16, generator=generator, dtype=torch.float64))
        self.fc1 = build_tt_linear(dense, (2, 4), (4, 4))
        # ``fc1.cores.`` / ``fc1`` に一致してはいけない、名前だけ近い parameter。
        self.fc1.register_parameter("cores_scale", nn.Parameter(torch.ones(1, dtype=torch.float64)))
        self.fc10 = nn.Linear(8, 8, dtype=torch.float64)
        self.fc1_aux = nn.Linear(8, 8, dtype=torch.float64)
        self.fc2 = nn.Linear(8, 4, dtype=torch.float64)

    def forward(self, x):
        h = self.fc1(x) * self.fc1.cores_scale
        return self.fc2(self.fc1_aux(self.fc10(h)))


class _TTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        conv1 = nn.Conv2d(1, 4, 3, dtype=torch.float64)
        conv2 = nn.Conv2d(4, 8, 3, dtype=torch.float64)
        self.conv1 = conv1
        self.bn = nn.BatchNorm2d(4, dtype=torch.float64)
        self.conv2 = build_tt_conv2d(conv2, (2, 4), (6, 6))
        self.conv20 = nn.Conv2d(8, 8, 1, dtype=torch.float64)
        self.fc = nn.Linear(8, 3, dtype=torch.float64)

    def forward(self, x):
        h = self.conv2(torch.relu(self.bn(self.conv1(x))))
        return self.fc(self.conv20(h).mean(dim=(2, 3)))


def _flags(model):
    return {name: parameter.requires_grad for name, parameter in model.named_parameters()}


def _all_names(model):
    return [name for name, _ in model.named_parameters()]


# ---------------------------------------------------------------------------
# set_trainable_parameter_prefixes
# ---------------------------------------------------------------------------


def test_fc1_cores_prefix_enables_only_fc1_cores_and_freezes_the_rest():
    model = _TTMLP()

    matched = set_trainable_parameter_prefixes(model, ["fc1.cores."])

    assert matched == ("fc1.cores.0", "fc1.cores.1")
    assert isinstance(matched, tuple)
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad is (name in matched), name


def test_conv2_cores_prefix_enables_only_conv2_cores():
    model = _TTCNN()

    matched = set_trainable_parameter_prefixes(model, ("conv2.cores.",))

    assert matched == tuple(f"conv2.cores.{k}" for k in range(len(model.conv2.cores)))
    assert {name for name, flag in _flags(model).items() if flag} == set(matched)
    assert not model.conv2.bias.requires_grad
    assert not model.conv20.weight.requires_grad


def test_matched_names_follow_registration_order_regardless_of_prefix_order():
    model = _TTMLP()

    matched = set_trainable_parameter_prefixes(model, ["fc2.", "fc1.cores."])

    assert matched == ("fc1.cores.0", "fc1.cores.1", "fc2.weight", "fc2.bias")
    assert list(matched) == [name for name in _all_names(model) if name in matched]


def test_multiple_prefixes_union_and_duplicates_are_harmless():
    model = _TTMLP()

    matched = set_trainable_parameter_prefixes(model, ["fc1.cores.", "fc2.bias", "fc2.bias", "fc1.cores."])

    assert matched == ("fc1.cores.0", "fc1.cores.1", "fc2.bias")
    assert {name for name, flag in _flags(model).items() if flag} == set(matched)


def test_prefix_without_trailing_dot_respects_module_boundaries():
    model = _TTMLP()

    matched = set_trainable_parameter_prefixes(model, ["fc1"])

    assert set(matched) == {"fc1.cores.0", "fc1.cores.1", "fc1.bias", "fc1.cores_scale"}
    assert not any(name.startswith(("fc10", "fc1_aux")) for name in matched)
    assert not model.fc10.weight.requires_grad and not model.fc1_aux.weight.requires_grad


def test_prefix_with_trailing_dot_does_not_match_similar_names():
    model = _TTMLP()

    matched = set_trainable_parameter_prefixes(model, ["fc1.cores."])

    assert "fc1.cores_scale" not in matched
    assert not model.fc1.cores_scale.requires_grad
    assert not model.fc10.weight.requires_grad


def test_dotless_prefix_matches_a_whole_parameter_list_but_not_similar_names():
    model = _TTMLP()

    matched = set_trainable_parameter_prefixes(model, ["fc1.cores"])

    assert matched == ("fc1.cores.0", "fc1.cores.1")  # "fc1.cores_scale" は含まない


def test_parameter_name_itself_is_a_valid_prefix():
    model = _TTMLP()

    matched = set_trainable_parameter_prefixes(model, ["fc2.bias"])

    assert matched == ("fc2.bias",)


@pytest.mark.parametrize("prefix", ["fc", "fc1.cor", "fc1.cores.00", "missing.", "FC1."])
def test_partial_word_or_unknown_prefix_matches_nothing(prefix):
    model = _TTMLP()
    flags_before = _flags(model)

    with pytest.raises(ValueError, match="一致する parameter がありません"):
        set_trainable_parameter_prefixes(model, [prefix])

    assert _flags(model) == flags_before  # 失敗しても何も凍結しない


def test_no_match_error_lists_available_names():
    model = _TTMLP()

    with pytest.raises(ValueError, match="fc1.cores.0"):
        set_trainable_parameter_prefixes(model, ["nope."])


def test_no_match_with_freeze_others_false_also_leaves_model_untouched():
    model = _TTMLP()
    model.fc2.weight.requires_grad_(False)
    flags_before = _flags(model)

    with pytest.raises(ValueError):
        set_trainable_parameter_prefixes(model, ["nope."], freeze_others=False)

    assert _flags(model) == flags_before


@pytest.mark.parametrize("prefixes", [[], ()])
def test_empty_prefix_sequence_is_rejected(prefixes):
    model = _TTMLP()
    flags_before = _flags(model)

    with pytest.raises(ValueError, match="空"):
        set_trainable_parameter_prefixes(model, prefixes)

    assert _flags(model) == flags_before


@pytest.mark.parametrize("prefixes", [[""], ["fc1.cores.", ""], ("",)])
def test_empty_string_prefix_is_rejected_before_any_change(prefixes):
    model = _TTMLP()
    flags_before = _flags(model)

    with pytest.raises(ValueError, match="空文字列"):
        set_trainable_parameter_prefixes(model, prefixes)

    assert _flags(model) == flags_before


@pytest.mark.parametrize(
    "prefixes",
    ["fc1.cores.", b"fc1.", {"fc1.cores."}, None, 3, True, (p for p in ["fc1."]), np.array(["fc1."])],
)
def test_non_sequence_prefixes_are_rejected(prefixes):
    model = _TTMLP()
    flags_before = _flags(model)

    with pytest.raises(TypeError, match="prefixes"):
        set_trainable_parameter_prefixes(model, prefixes)

    assert _flags(model) == flags_before


@pytest.mark.parametrize("bad", [None, 1, True, b"fc1.", ["fc1."]])
def test_non_string_prefix_elements_are_rejected(bad):
    model = _TTMLP()
    flags_before = _flags(model)

    with pytest.raises(TypeError, match="prefix"):
        set_trainable_parameter_prefixes(model, ["fc1.cores.", bad])

    assert _flags(model) == flags_before


@pytest.mark.parametrize("flag", [1, 0, None, "yes", np.bool_(True)])
def test_freeze_others_must_be_a_bool(flag):
    with pytest.raises(TypeError, match="freeze_others"):
        set_trainable_parameter_prefixes(_TTMLP(), ["fc1."], freeze_others=flag)


def test_model_must_be_a_module():
    with pytest.raises(TypeError, match="nn.Module"):
        set_trainable_parameter_prefixes({"fc1.weight": 1}, ["fc1."])


def test_freeze_others_false_leaves_other_flags_unchanged():
    model = _TTMLP()
    model.fc2.weight.requires_grad_(False)
    model.fc10.weight.requires_grad_(False)
    flags_before = _flags(model)

    matched = set_trainable_parameter_prefixes(model, ["fc10.weight"], freeze_others=False)

    expected = dict(flags_before)
    expected["fc10.weight"] = True
    assert matched == ("fc10.weight",)
    assert _flags(model) == expected


def test_freeze_others_true_unfreezes_matched_parameters_that_were_frozen():
    model = _TTMLP()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    set_trainable_parameter_prefixes(model, ["fc1.cores."])

    assert model.fc1.cores[0].requires_grad and model.fc1.cores[1].requires_grad
    assert not model.fc2.weight.requires_grad


def test_calling_twice_switches_scope_without_accumulating():
    model = _TTMLP()

    set_trainable_parameter_prefixes(model, ["fc1.cores."])
    set_trainable_parameter_prefixes(model, ["fc2."])

    assert {name for name, flag in _flags(model).items() if flag} == {"fc2.weight", "fc2.bias"}


def test_parameter_values_buffers_and_training_state_are_unchanged():
    model = _TTCNN().train()
    model.bn.eval()
    state_before = copy.deepcopy(model.state_dict())
    training_before = {name: module.training for name, module in model.named_modules()}

    set_trainable_parameter_prefixes(model, ["conv2.cores."])

    for name, value in model.state_dict().items():
        assert torch.equal(value, state_before[name]), name
    assert {name: module.training for name, module in model.named_modules()} == training_before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_parameter_objects_are_the_same_instances_after_scoping():
    model = _TTMLP()
    ids_before = {name: id(parameter) for name, parameter in model.named_parameters()}

    set_trainable_parameter_prefixes(model, ["fc1.cores."])

    assert {name: id(parameter) for name, parameter in model.named_parameters()} == ids_before


# ---------------------------------------------------------------------------
# trainable_named_parameters
# ---------------------------------------------------------------------------


def test_trainable_named_parameters_returns_only_trainable_in_registration_order():
    model = _TTMLP()
    set_trainable_parameter_prefixes(model, ["fc2.", "fc1.cores."])

    trainable = trainable_named_parameters(model)

    assert isinstance(trainable, list)
    assert [name for name, _ in trainable] == ["fc1.cores.0", "fc1.cores.1", "fc2.weight", "fc2.bias"]
    assert all(isinstance(parameter, nn.Parameter) and parameter.requires_grad for _, parameter in trainable)


def test_trainable_named_parameters_returns_the_models_own_parameters_not_copies():
    model = _TTMLP()
    set_trainable_parameter_prefixes(model, ["fc1.cores."])

    trainable = dict(trainable_named_parameters(model))

    assert trainable["fc1.cores.0"] is model.fc1.cores[0]
    assert trainable["fc1.cores.1"] is model.fc1.cores[1]


def test_trainable_named_parameters_is_the_optimizers_parameter_set():
    model = _TTMLP()
    matched = set_trainable_parameter_prefixes(model, ["fc1.cores.", "fc2.bias"])
    trainable = trainable_named_parameters(model)

    optimizer = torch.optim.SGD([parameter for _, parameter in trainable], lr=0.1)

    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    requires_grad_ids = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert optimizer_ids == requires_grad_ids
    assert tuple(name for name, _ in trainable) == matched


def test_trainable_named_parameters_does_not_change_flags_or_values():
    model = _TTMLP()
    set_trainable_parameter_prefixes(model, ["fc1.cores."])
    flags_before = _flags(model)
    state_before = copy.deepcopy(model.state_dict())

    trainable_named_parameters(model)

    assert _flags(model) == flags_before
    for name, value in model.state_dict().items():
        assert torch.equal(value, state_before[name])


def test_trainable_named_parameters_all_frozen_gives_empty_list_and_all_trainable_gives_all():
    model = _TTMLP()
    assert [name for name, _ in trainable_named_parameters(model)] == _all_names(model)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert trainable_named_parameters(model) == []


def test_trainable_named_parameters_lists_shared_parameter_once():
    model = nn.Module()
    model.a = nn.Linear(3, 3)
    model.b = model.a  # 同じ module を別名で登録

    assert [name for name, _ in trainable_named_parameters(model)] == ["a.weight", "a.bias"]


def test_trainable_named_parameters_requires_a_module():
    with pytest.raises(TypeError, match="nn.Module"):
        trainable_named_parameters([nn.Parameter(torch.ones(1))])


# ---------------------------------------------------------------------------
# parameter_change_summary
# ---------------------------------------------------------------------------


def _train_one_step(model, lr=0.1):
    trainable = [parameter for _, parameter in trainable_named_parameters(model)]
    optimizer = torch.optim.SGD(trainable, lr=lr)
    x = torch.randn(5, 16, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    loss = model(x).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_summary_schema_and_registration_order():
    model = _TTMLP()
    before = copy.deepcopy(model.state_dict())

    rows = parameter_change_summary(before, model)

    assert [row["name"] for row in rows] == _all_names(model)
    assert all(list(row) == ["name", "requires_grad", "numel", "exactly_equal", "max_abs_change"] for row in rows)
    for row, (name, parameter) in zip(rows, model.named_parameters()):
        assert row["numel"] == parameter.numel() and type(row["numel"]) is int
        assert row["requires_grad"] is parameter.requires_grad
        assert type(row["exactly_equal"]) is bool and type(row["max_abs_change"]) is float


def test_summary_of_untouched_model_reports_everything_equal():
    model = _TTMLP()

    rows = parameter_change_summary(copy.deepcopy(model.state_dict()), model)

    assert all(row["exactly_equal"] and row["max_abs_change"] == 0.0 for row in rows)


def test_summary_records_trainable_changes_and_shows_frozen_parameters_unchanged():
    model = _TTMLP()
    set_trainable_parameter_prefixes(model, ["fc1.cores."])
    before = copy.deepcopy(model.state_dict())

    _train_one_step(model)
    rows = {row["name"]: row for row in parameter_change_summary(before, model)}

    for name in ("fc1.cores.0", "fc1.cores.1"):
        assert rows[name]["requires_grad"] is True
        assert rows[name]["exactly_equal"] is False
        expected = float((dict(model.named_parameters())[name].detach() - before[name]).abs().max())
        assert rows[name]["max_abs_change"] == pytest.approx(expected, rel=1e-14)
        assert rows[name]["max_abs_change"] > 0.0
    frozen = [row for row in rows.values() if not row["requires_grad"]]
    assert {row["name"] for row in frozen} == set(rows) - {"fc1.cores.0", "fc1.cores.1"}
    assert all(row["exactly_equal"] and row["max_abs_change"] == 0.0 for row in frozen)


def test_summary_detects_a_change_in_a_frozen_parameter():
    model = _TTMLP()
    set_trainable_parameter_prefixes(model, ["fc1.cores."])
    before = copy.deepcopy(model.state_dict())

    with torch.no_grad():
        model.fc2.bias[1] += 0.25  # 凍結 parameter を誤って更新した状況

    rows = {row["name"]: row for row in parameter_change_summary(before, model)}
    violations = [name for name, row in rows.items() if not row["requires_grad"] and not row["exactly_equal"]]
    assert violations == ["fc2.bias"]
    assert rows["fc2.bias"]["max_abs_change"] == pytest.approx(0.25, rel=1e-14)


def test_summary_detects_one_ulp_change_in_float32():
    linear = nn.Linear(2, 2)
    before = copy.deepcopy(linear.state_dict())
    with torch.no_grad():
        linear.weight[0, 0] = torch.nextafter(linear.weight[0, 0], torch.tensor(10.0))

    rows = {row["name"]: row for row in parameter_change_summary(before, linear)}

    assert rows["weight"]["exactly_equal"] is False
    assert 0.0 < rows["weight"]["max_abs_change"] < 1e-5
    assert rows["bias"]["exactly_equal"] is True


def test_summary_accepts_buffers_in_state_and_parameter_only_state():
    model = _TTCNN()
    full_state = copy.deepcopy(model.state_dict())  # BatchNorm の running_mean / num_batches_tracked を含む
    parameter_only = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}

    rows_full = parameter_change_summary(full_state, model)
    rows_only = parameter_change_summary(parameter_only, model)

    assert rows_full == rows_only
    assert [row["name"] for row in rows_full] == _all_names(model)
    assert not any("running" in row["name"] for row in rows_full)


def test_summary_for_conv_model_after_cores_only_training():
    model = _TTCNN()
    set_trainable_parameter_prefixes(model, ["conv2.cores."])
    before = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.SGD([p for _, p in trainable_named_parameters(model)], lr=0.1)
    model.train()
    x = torch.randn(4, 1, 8, 8, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
    model(x).square().mean().backward()
    optimizer.step()

    rows = parameter_change_summary(before, model)

    changed = {row["name"] for row in rows if not row["exactly_equal"]}
    assert changed == {name for name in _all_names(model) if name.startswith("conv2.cores.")}


def test_summary_does_not_modify_state_model_or_flags():
    model = _TTMLP()
    set_trainable_parameter_prefixes(model, ["fc1.cores."])
    before = copy.deepcopy(model.state_dict())
    snapshot = copy.deepcopy(before)
    flags_before = _flags(model)
    training_before = {name: module.training for name, module in model.named_modules()}

    parameter_change_summary(before, model)

    for name in before:
        assert torch.equal(before[name], snapshot[name])
        assert torch.equal(model.state_dict()[name], snapshot[name])
    assert _flags(model) == flags_before
    assert {name: module.training for name, module in model.named_modules()} == training_before


def test_summary_records_nan_and_inf_without_raising():
    model = nn.Linear(2, 2, dtype=torch.float64)
    before = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        model.weight[0, 0] = math.nan
        model.bias[0] = math.inf

    rows = {row["name"]: row for row in parameter_change_summary(before, model)}

    assert rows["weight"]["exactly_equal"] is False and math.isnan(rows["weight"]["max_abs_change"])
    assert rows["bias"]["exactly_equal"] is False and rows["bias"]["max_abs_change"] == math.inf


def test_summary_of_zero_element_parameter():
    model = nn.Module()
    model.empty = nn.Parameter(torch.empty(0, 3))

    rows = {row["name"]: row for row in parameter_change_summary(copy.deepcopy(model.state_dict()), model)}

    assert rows["empty"]["numel"] == 0
    assert rows["empty"]["exactly_equal"] is True and rows["empty"]["max_abs_change"] == 0.0


def test_summary_of_model_without_parameters_is_empty():
    assert parameter_change_summary({}, nn.Identity()) == []


def test_summary_rows_can_be_passed_to_pandas_directly():
    pd = pytest.importorskip("pandas")
    model = _TTMLP()
    before = copy.deepcopy(model.state_dict())

    frame = pd.DataFrame(parameter_change_summary(before, model))

    assert list(frame.columns) == ["name", "requires_grad", "numel", "exactly_equal", "max_abs_change"]
    assert len(frame) == len(_all_names(model))
    assert frame["requires_grad"].dtype == bool and frame["exactly_equal"].dtype == bool


def test_summary_rejects_missing_parameter():
    model = _TTMLP()
    before = copy.deepcopy(model.state_dict())
    del before["fc2.bias"]

    with pytest.raises(ValueError, match="不足.*fc2.bias"):
        parameter_change_summary(before, model)


def test_summary_rejects_unknown_extra_key():
    model = _TTMLP()
    before = copy.deepcopy(model.state_dict())
    before["fc3.weight"] = torch.zeros(1, dtype=torch.float64)

    with pytest.raises(ValueError, match="存在しないキー.*fc3.weight"):
        parameter_change_summary(before, model)


def test_summary_rejects_shape_mismatch():
    model = _TTMLP()
    before = copy.deepcopy(model.state_dict())
    before["fc2.weight"] = torch.zeros(4, 9, dtype=torch.float64)

    with pytest.raises(ValueError, match="shape.*fc2.weight|fc2.weight.*shape"):
        parameter_change_summary(before, model)


def test_summary_rejects_dtype_and_device_mismatch_without_implicit_conversion():
    model = _TTMLP()
    before = copy.deepcopy(model.state_dict())
    before["fc2.bias"] = before["fc2.bias"].to(torch.float32)
    with pytest.raises(TypeError, match="dtype"):
        parameter_change_summary(before, model)

    before = copy.deepcopy(model.state_dict())
    before["fc2.bias"] = torch.zeros(4, dtype=torch.float64, device="meta")
    with pytest.raises(ValueError, match="device"):
        parameter_change_summary(before, model)


def test_summary_rejects_invalid_argument_types():
    model = _TTMLP()
    with pytest.raises(TypeError, match="Mapping"):
        parameter_change_summary(list(model.state_dict().items()), model)
    with pytest.raises(TypeError, match="nn.Module"):
        parameter_change_summary(model.state_dict(), model.state_dict())
    before = copy.deepcopy(model.state_dict())
    before["fc2.bias"] = [0.0] * 4
    with pytest.raises(TypeError, match="torch.Tensor"):
        parameter_change_summary(before, model)


def _change_row(before: torch.Tensor, current: torch.Tensor) -> dict:
    """1 parameter のモデルで、学習前 state と現在値の比較結果を1行返す。"""
    model = nn.Module()
    # 整数 parameter は勾配を持てない。比較そのものは requires_grad に依存しない。
    requires_grad = current.is_floating_point() or current.is_complex()
    model.weight = nn.Parameter(current.detach().clone(), requires_grad=requires_grad)
    before_state = {"weight": before.detach().clone()}
    before_snapshot = before_state["weight"].clone()
    current_snapshot = model.weight.detach().clone()

    row = parameter_change_summary(before_state, model)[0]

    assert torch.equal(before_state["weight"], before_snapshot)
    assert torch.equal(model.weight.detach(), current_snapshot)
    return row


# ---------------------------------------------------------------------------
# complex / int64 / inf の差分（P2-2, P2-3, P2-4）
# ---------------------------------------------------------------------------


def test_complex64_imaginary_only_change_is_measured():
    before = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)
    current = torch.tensor([1 + 5j, 3 + 4j], dtype=torch.complex64)

    row = _change_row(before, current)

    assert row["exactly_equal"] is False
    assert row["max_abs_change"] == 3.0
    assert type(row["max_abs_change"]) is float


def test_complex128_real_and_imaginary_changes_use_complex_magnitude():
    before = torch.tensor([1 + 2j, 0 + 0j], dtype=torch.complex128)
    current = torch.tensor([1 + 5j, 3 + 4j], dtype=torch.complex128)

    row = _change_row(before, current)

    # |3j| = 3、|3+4j| = 5。最大は 5。
    assert row["exactly_equal"] is False
    assert row["max_abs_change"] == 5.0
    assert type(row["max_abs_change"]) is float


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_unchanged_complex_parameter_reports_zero_change(dtype):
    value = torch.tensor([1 + 2j, -4 + 0.5j, 0 + 0j], dtype=dtype)

    row = _change_row(value, value.clone())

    assert row["exactly_equal"] is True
    assert row["max_abs_change"] == 0.0
    assert type(row["max_abs_change"]) is float


def test_int64_one_above_two_pow_62_is_a_change_of_one():
    before = torch.tensor([2**62, 0], dtype=torch.int64)
    current = torch.tensor([2**62 + 1, 0], dtype=torch.int64)

    row = _change_row(before, current)

    assert row["exactly_equal"] is False
    assert row["max_abs_change"] == 1.0
    assert type(row["max_abs_change"]) is float


def test_int64_near_maximum_keeps_a_unit_difference():
    highest = torch.iinfo(torch.int64).max
    before = torch.tensor([highest - 1, 4], dtype=torch.int64)
    current = torch.tensor([highest, 4], dtype=torch.int64)

    row = _change_row(before, current)

    assert row["exactly_equal"] is False
    assert row["max_abs_change"] == 1.0


def test_int64_near_minimum_keeps_a_unit_difference():
    lowest = torch.iinfo(torch.int64).min
    before = torch.tensor([lowest], dtype=torch.int64)
    current = torch.tensor([lowest + 1], dtype=torch.int64)

    row = _change_row(before, current)

    assert row["exactly_equal"] is False
    assert row["max_abs_change"] == 1.0


def test_int64_minimum_versus_maximum_is_a_finite_python_float():
    lowest = torch.iinfo(torch.int64).min
    highest = torch.iinfo(torch.int64).max

    row = _change_row(
        torch.tensor([lowest], dtype=torch.int64),
        torch.tensor([highest], dtype=torch.int64),
    )

    assert row["exactly_equal"] is False
    assert row["max_abs_change"] == float(int(highest) - int(lowest))
    assert math.isfinite(row["max_abs_change"])
    assert row["max_abs_change"] > 0.0


def test_unchanged_int64_parameter_reports_zero_change():
    value = torch.tensor([2**62, -(2**62), torch.iinfo(torch.int64).min, 0], dtype=torch.int64)

    row = _change_row(value, value.clone())

    assert row["exactly_equal"] is True
    assert row["max_abs_change"] == 0.0


def test_int64_selects_the_largest_element_difference():
    before = torch.tensor([2**62, 10, -(2**61)], dtype=torch.int64)
    current = torch.tensor([2**62 + 1, 10, -(2**61) - 5], dtype=torch.int64)

    row = _change_row(before, current)

    assert row["exactly_equal"] is False
    assert row["max_abs_change"] == 5.0


def test_int64_exactly_equal_and_max_abs_change_do_not_contradict():
    cases = [
        (torch.tensor([2**62], dtype=torch.int64), torch.tensor([2**62], dtype=torch.int64)),
        (torch.tensor([2**62], dtype=torch.int64), torch.tensor([2**62 + 1], dtype=torch.int64)),
        (
            torch.tensor([torch.iinfo(torch.int64).min], dtype=torch.int64),
            torch.tensor([torch.iinfo(torch.int64).max], dtype=torch.int64),
        ),
    ]
    for before, current in cases:
        row = _change_row(before, current)
        if row["exactly_equal"]:
            assert row["max_abs_change"] == 0.0
        else:
            assert row["max_abs_change"] > 0.0
            assert not math.isnan(row["max_abs_change"])


@pytest.mark.parametrize(
    "value",
    [
        torch.tensor([math.inf, math.inf], dtype=torch.float64),
        torch.tensor([-math.inf], dtype=torch.float32),
        torch.tensor([math.inf, -math.inf, 1.5], dtype=torch.float64),
    ],
)
def test_identical_infinities_report_zero_change(value):
    row = _change_row(value, value.clone())

    assert row["exactly_equal"] is True
    assert row["max_abs_change"] == 0.0
    assert not math.isnan(row["max_abs_change"])


def test_positive_inf_to_negative_inf_is_an_infinite_change():
    row = _change_row(
        torch.tensor([math.inf, 0.0], dtype=torch.float64),
        torch.tensor([-math.inf, 0.0], dtype=torch.float64),
    )

    assert row["exactly_equal"] is False
    assert row["max_abs_change"] == math.inf


def test_finite_to_positive_inf_is_an_infinite_change():
    row = _change_row(
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        torch.tensor([1.0, math.inf], dtype=torch.float32),
    )

    assert row["exactly_equal"] is False
    assert row["max_abs_change"] == math.inf
    assert not math.isnan(row["max_abs_change"])


def test_positive_inf_to_finite_is_an_infinite_change():
    row = _change_row(
        torch.tensor([math.inf, -3.0], dtype=torch.float64),
        torch.tensor([4.0, -3.0], dtype=torch.float64),
    )

    assert row["exactly_equal"] is False
    assert row["max_abs_change"] == math.inf


def test_same_sign_inf_does_not_turn_a_finite_change_into_nan():
    row = _change_row(
        torch.tensor([math.inf, 1.0, -math.inf], dtype=torch.float64),
        torch.tensor([math.inf, 1.25, -math.inf], dtype=torch.float64),
    )

    assert row["exactly_equal"] is False
    assert row["max_abs_change"] == pytest.approx(0.25)
    assert not math.isnan(row["max_abs_change"])


def test_finite_float_change_is_unchanged_by_the_nonfinite_handling():
    before = torch.tensor([1.0, -2.5, 4.0], dtype=torch.float64)
    current = before.clone()
    current[1] = -2.25

    row = _change_row(before, current)

    assert row["exactly_equal"] is False
    assert row["max_abs_change"] == pytest.approx(0.25)


def test_exactly_equal_float_parameters_have_zero_max_abs_change():
    values = [
        torch.tensor([1.0, -2.0, 3.5], dtype=torch.float32),
        torch.tensor([math.inf, 0.0, -math.inf], dtype=torch.float64),
        torch.tensor([-math.inf, -math.inf], dtype=torch.float64),
    ]
    for value in values:
        row = _change_row(value, value.clone())
        assert row["exactly_equal"] is True
        assert row["max_abs_change"] == 0.0
